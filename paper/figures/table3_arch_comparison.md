# Table 3 — U-Net vs SegResNet, 5-fold CV (n=187), with Wilcoxon's r effect size

Paired Wilcoxon signed-rank test, Bonferroni-corrected (×4 OARs).
Effect size r = |Z|/sqrt(N). Cohen interpretation: 0.10 small, 0.30 medium, 0.50 large.
Δ = SegResNet − U-Net.

## DSC (higher is better)

| OAR | n | U-Net (median) | SegResNet (median) | Δ | p (Bonf.) | r | Effect size |
|---|---:|---:|---:|---:|---:|---:|:--:|
| Left lung | 187 | 0.881 | 0.933 | +0.051 | <0.001 \*\*\* | 0.833 | **large** |
| Right lung | 186 | 0.904 | 0.941 | +0.037 | <0.001 \*\*\* | 0.781 | **large** |
| Heart | 186 | 0.801 | 0.885 | +0.084 | <0.001 \*\*\* | 0.854 | **large** |
| Esophagus | 184 | 0.409 | 0.666 | +0.257 | <0.001 \*\*\* | 0.838 | **large** |

## HD95 (mm) (lower is better)

| OAR | n | U-Net (median) | SegResNet (median) | Δ | p (Bonf.) | r | Effect size |
|---|---:|---:|---:|---:|---:|---:|:--:|
| Left lung | 184 | 8.3 | 3.0 | -5.3 | <0.001 \*\*\* | 0.689 | **large** |
| Right lung | 180 | 7.0 | 2.4 | -4.6 | <0.001 \*\*\* | 0.700 | **large** |
| Heart | 183 | 9.9 | 5.7 | -4.3 | <0.001 \*\*\* | 0.694 | **large** |
| Esophagus | 183 | 9.0 | 3.7 | -5.3 | <0.001 \*\*\* | 0.521 | **large** |

## Surface DSC @2mm (higher is better)

| OAR | n | U-Net (median) | SegResNet (median) | Δ | p (Bonf.) | r | Effect size |
|---|---:|---:|---:|---:|---:|---:|:--:|
| Left lung | 187 | 0.793 | 0.919 | +0.126 | <0.001 \*\*\* | 0.846 | **large** |
| Right lung | 186 | 0.834 | 0.943 | +0.109 | <0.001 \*\*\* | 0.802 | **large** |
| Heart | 186 | 0.687 | 0.845 | +0.159 | <0.001 \*\*\* | 0.865 | **large** |
| Esophagus | 184 | 0.631 | 0.882 | +0.251 | <0.001 \*\*\* | 0.786 | **large** |

\*\*\* p < 0.001 Bonferroni-corrected. All effect sizes >= 0.50 (Cohen 'large').

## Mean DSC across the 4 OARs (per patient)

- **U-Net**: median = 0.731, mean = 0.690 ± 0.157 (n=187)
- **SegResNet**: median = 0.842, mean = 0.813 ± 0.109 (n=187)

**Mean-DSC paired comparison (n=187)**: median Δ = +0.100, p < 0.001 (Wilcoxon), r = 0.865 (**large**)
