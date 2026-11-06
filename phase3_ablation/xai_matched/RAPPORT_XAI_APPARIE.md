# XAI à couches appariées : E3 (U-Net 18,8 M) vs E2 (SegResNet 18,8 M)

Date : 06/11/2026. Seg-Grad-CAM 3D, 187 patients (validation croisée en 5 plis, plis de phase 2), checkpoints `best.pt` de phase 2.

**Scripts :**
- calcul : `phase3_ablation/xai_matched/xai_matched_kaggle.py` ;
- lanceur : `launch_xai.py` ;
- analyse : `analyse/analyse_xai_matched.py`.

**Sorties brutes :** `results/phase3_ablation/xai_matched/`.

**Environnement :** Kaggle T4, torch 2.10.0+cu128, MONAI 1.5.2, code au commit `0a36a1e`. Durée : 14 min (E3, plis 0–3) et 28 min (E2 plis 0–4 + E3 pli 4).

## Pourquoi refaire le XAI

L'analyse précédente utilisait `target_layer="auto"` (dernière Conv3d). Elle mesurait donc deux choses différentes :
- pour l'U-Net, le **goulot** (Conv3d 384→832, résolution 1/16) ;
- pour SegResNet, la **tête de sortie 1×1** (`conv_final`).

Les deux modèles n'étaient pas comparés au même niveau.

## Méthode

On mesure quatre niveaux dans une seule passe par organe :

| Niveau | U-Net (E3) | SegResNet (E2) |
|---|---|---|
| enc_profond | bloc du fond, 832 canaux, 1/16 | `down_layers.3`, 256 canaux, 1/8 |
| dec_demi | `model.1` (skip + up), 96 canaux, 1/2 | `up_layers.1`, 64 canaux, 1/2 |
| dec_pre_tete | `model.1`, 96 canaux, 1/2 (même module que dec_demi) | `up_layers.2`, 32 canaux, 1/1 |
| sortie | `model.2` (logits) | `conv_final` (logits) |

- **Recadrage :** 128 × 128 × 64, centré sur la boîte englobante des 4 OAR de référence. Le cœur et l'œsophage sont entièrement dans le champ (fraction médiane 1,00). Les poumons y sont à 74–79 % en médiane.
- **Tests statistiques :**
  - Wilcoxon apparié par patient (E2 − E3) ;
  - correction de Holm par métrique (16 tests) ;
  - IC 95 % bootstrap de la médiane des différences ;
  - r rang-biserial pour la taille d'effet.
- **Test d'Adebayo :** 1 patient par pli, œsophage. Randomisation en cascade de toutes les Conv3d et ConvTranspose3d (9 couches pour l'U-Net, 32 pour SegResNet), mesurée par SSIM et par corrélation de Spearman.

## Résultats observés

**1. L'ancienne analyse est reproduite aux mêmes couches.**
- E2 au niveau sortie : in-organ médian 0,94 / 0,95 / 0,88 / 0,75 (poumon G / poumon D / cœur / œsophage). L'ancienne analyse donnait 0,93 / 0,94 / 0,87 / 0,76.
- E3 au niveau enc_profond : 0,25 / 0,12 / 0,25 / 0,00, contre 0,03 / 0,04 / 0,03 / 0,00 auparavant. L'écart vient du recadrage et du point de mesure (après normalisation et activation).
- Le contraste ancien « SegResNet ≈ 0,9 vs U-Net ≈ 0,0 » s'explique donc en grande partie par le choix des couches.

**2. In-organ ratio (médiane E3 → E2 ; différence appariée et IC 95 %)**

| Niveau | Poumon G | Poumon D | Cœur | Œsophage |
|---|---|---|---|---|
| enc_profond | 0,25 → 0,46 (+0,19 [0,15 ; 0,21]) | 0,12 → 0,61 (+0,40) | 0,26 → 0,72 (+0,38) | 0,00 → 0,00 |
| dec_demi | 0,88 → 0,93 (+0,04) | 0,89 → 0,90 (n.s., p_Holm = 0,51) | 0,81 → 0,70 (−0,08) | 0,42 → 0,02 (−0,29 [−0,36 ; −0,25]) |
| dec_pre_tete | 0,88 → 0,90 (+0,02) | 0,89 → 0,92 (+0,03) | 0,81 → 0,86 (+0,04) | 0,41 → 0,16 (−0,19 [−0,23 ; −0,14]) |
| sortie | 0,90 → 0,94 (+0,03) | 0,90 → 0,95 (+0,03) | 0,82 → 0,88 (+0,05) | 0,44 → 0,76 (+0,25 [0,23 ; 0,27]) |

Toutes les différences ont p_Holm < 0,001, sauf poumon D au niveau dec_demi. Le fichier complet `analyse/xai_stats_apparies.csv` donne aussi l'activation-Dice, l'entropie, la distance des centroïdes et le gain par rapport à une carte uniforme.

**3. Pointing accuracy :** la médiane vaut 1 pour les deux modèles aux niveaux décodeur et sortie. Elle est saturée et ne permet pas de discriminer les modèles.

**4. Test d'Adebayo :** après randomisation de la seule couche de sortie, la SSIM médiane est ≤ 0,06 à tous les niveaux, pour les deux modèles. La SSIM maximale observée sur toute la cascade est ≤ 0,32. Les cartes dépendent donc des poids appris : le test est réussi.

**5. Couverture :** aucun organe hors champ. Il manque seulement 1 à 3 valeurs par cellule (organe non prédit dans le recadrage). L'ancienne analyse en avait 19 à 30.

## Résultats de la littérature

- Adebayo et al. (NeurIPS 2018) : le test de randomisation des poids.
- Binder et al. (CVPR 2023) : réussir ce test ne suffit pas à classer les méthodes d'explication.
- Arun et al. (*Radiology: AI* 2021) : les cartes de saillance en imagerie médicale échouent souvent à au moins un critère de fiabilité.
- Saporta et al. (*Nat Mach Intell* 2022) : Grad-CAM localise moins bien les petites structures.
- Strijbis et al. (*Comput Biol Med* 2025) : XAI sur un U-Net 3D pour les organes à risque ORL. C'est l'étude la plus proche de celle-ci.

Références obtenues avec Consensus. Les articles fondateurs de Grad-CAM et Seg-Grad-CAM sont à vérifier.

## Comparaison

Aux couches appariées du décodeur et de la sortie, les deux modèles concentrent 80 à 95 % de l'attribution dans les poumons et le cœur. Les écarts entre modèles restent ≤ 0,05 : ils sont significatifs, mais petits.

Pour l'œsophage, le sens de l'écart dépend du niveau :
- **E3 est plus focalisé dans le décodeur** (0,41 contre 0,02–0,16) ;
- **E2 est plus focalisé à la sortie** (0,76 contre 0,44).

Au niveau enc_profond, E2 est plus focalisé, mais les résolutions diffèrent (1/8 contre 1/16).

## Interprétations possibles (non démontrées)

- La conclusion « les explications de SegResNet sont plus ancrées anatomiquement » n'est **pas soutenue** : elle dépend de la couche choisie.
- Pour l'œsophage, E2 segmente mieux (DSC médian 0,68 contre 0,50) alors que l'attribution de son décodeur est moins localisée sur l'organe. La localisation Seg-Grad-CAM ne suit donc pas la qualité de segmentation. Hypothèse : SegResNet s'appuie davantage sur le contexte autour de l'œsophage. Ce n'est pas testé.

## Limites

- Les résolutions ne sont pas strictement appariées, à cause de la structure des réseaux : U-Net à 4 sous-échantillonnages et tête en ConvTranspose, SegResNet à 3. Pour l'U-Net, dec_demi et dec_pre_tete sont le même module.
- Un seul recadrage de 128 × 128 × 64 : les poumons ne sont que partiellement couverts.
- Seg-Grad-CAM uniquement, sans comparaison à une autre méthode (Integrated Gradients, occlusion, etc.).
- Une seule graine d'entraînement.
- Adebayo sur 1 patient par pli, et pour l'œsophage seulement. La cible de Seg-Grad-CAM est le masque prédit, qui change dès que la sortie est randomisée : cela rend le test en partie facile à réussir (cf. Binder et al.).
- IRM diagnostiques ; pas de validation externe.
