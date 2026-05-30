# Table 3 — Architecture comparison: U-Net vs SegResNet (5-fold CV, n=187 patients)

Paired Wilcoxon signed-rank test with Bonferroni correction (×4 OARs).
Values are median [IQR]. Δ = SegResNet − U-Net.

## DSC (higher is better)

| OAR | n | U-Net | SegResNet | Δ | p (Bonf.) | Sig. |
|---|---:|---:|---:|---:|---:|:--:|
| Left lung | 187 | 0.881 [0.801–0.925] | 0.933 [0.891–0.957] | +0.051 | <0.001 | \*\*\* |
| Right lung | 187 | 0.904 [0.818–0.940] | 0.941 [0.908–0.964] | +0.037 | <0.001 | \*\*\* |
| Heart | 187 | 0.801 [0.661–0.864] | 0.885 [0.820–0.923] | +0.084 | <0.001 | \*\*\* |
| Esophagus | 187 | 0.409 [0.265–0.530] | 0.666 [0.543–0.755] | +0.257 | <0.001 | \*\*\* |

## HD95 (mm) (lower is better)

| OAR | n | U-Net | SegResNet | Δ | p (Bonf.) | Sig. |
|---|---:|---:|---:|---:|---:|:--:|
| Left lung | 187 | 8.3 [3.7–56.5] | 3.0 [2.0–8.3] | -5.3 | <0.001 | \*\*\* |
| Right lung | 187 | 7.0 [3.0–42.0] | 2.4 [1.4–5.9] | -4.6 | <0.001 | \*\*\* |
| Heart | 187 | 9.9 [5.1–30.9] | 5.7 [3.0–9.4] | -4.3 | <0.001 | \*\*\* |
| Esophagus | 183 | 9.0 [5.2–17.3] | 3.7 [2.3–9.4] | -5.3 | <0.001 | \*\*\* |

## Surface DSC @2mm (higher is better)

| OAR | n | U-Net | SegResNet | Δ | p (Bonf.) | Sig. |
|---|---:|---:|---:|---:|---:|:--:|
| Left lung | 187 | 0.793 [0.643–0.894] | 0.919 [0.836–0.974] | +0.126 | <0.001 | \*\*\* |
| Right lung | 187 | 0.834 [0.705–0.927] | 0.943 [0.850–0.983] | +0.109 | <0.001 | \*\*\* |
| Heart | 187 | 0.687 [0.509–0.797] | 0.845 [0.739–0.922] | +0.159 | <0.001 | \*\*\* |
| Esophagus | 187 | 0.631 [0.450–0.743] | 0.882 [0.755–0.936] | +0.251 | <0.001 | \*\*\* |

\*\*\* p < 0.001 (Bonferroni-corrected for 4 simultaneous tests per metric)

## Mean DSC across the 4 OARs (per patient)

- **U-Net** : 0.731 [0.632–0.798] (mean 0.690 ± 0.157, n=187)
- **SegResNet** : 0.842 [0.776–0.888] (mean 0.813 ± 0.109, n=187)
