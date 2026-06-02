# Adebayo cascading-randomization — Kaggle dual-account orchestration

## What this is

Implementation of the Adebayo et al. (2018, NeurIPS) sanity check for
SEG-GRAD-CAM 3D saliency maps. For each Conv3d layer of the trained network,
working from the deepest layer back toward the input, we re-initialise the
weights and recompute the saliency map. The SSIM(original, randomised) curve
should decay monotonically — a flat curve means the saliency does NOT depend
on the learned weights (failed sanity check).

## File layout

```
src/xai/adebayo.py                           ← core algorithm
scripts/run_adebayo_analysis.py              ← Python runner (1 model x 1 fold)
scripts/process_adebayo_results.py           ← CSV aggregator
scripts/kaggle/
  adebayo_kaggle.py                          ← Kaggle entry point (Python)
  adebayo_kaggle.ipynb                       ← generated notebook
  make_adebayo_notebook.py                   ← builds the ipynb + kernel-metadata.json
  12_run_adebayo_batch.ps1                   ← PowerShell launcher (Windows)
  launch_adebayo_dual.sh                     ← Bash launcher (Linux/macOS/WSL)
  smoke_test_adebayo.sh                      ← local 1-patient smoke test
  ADEBAYO_README.md                          ← this file
paper/figures/generate_figure8_adebayo.py    ← SSIM decay figure (Pub 3)
```

## End-to-end workflow

### 1. Smoke test locally (optional but recommended — 5 min)

Validate the runner produces a non-empty CSV before paying for the 5–8h
Kaggle run.

```bash
# After at least one training fold has been run locally:
bash scripts/kaggle/smoke_test_adebayo.sh unet 0 runs/unet_fold0/best.pt
```

Output: `results/adebayo_smoke/unet_fold0_adebayo_full.csv`.

### 2. Set up Kaggle tokens (one-time)

The dual-account launcher expects two backup tokens:

```bash
# Compte 1 (U-Net training notebooks)
cp ~/.kaggle/kaggle.json ~/.kaggle/kaggle.json.compte1

# Switch to compte 2 in your browser, download the new kaggle.json,
# place it at the standard location, then:
cp ~/.kaggle/kaggle.json ~/.kaggle/kaggle.json.compte2
```

Both files must be `chmod 600`.

### 3. Launch the dual run

**On Linux / macOS / WSL / Git Bash:**

```bash
bash scripts/kaggle/launch_adebayo_dual.sh
```

**On Windows PowerShell:**

```powershell
.\scripts\kaggle\12_run_adebayo_batch.ps1
```

Both launchers do the same thing:

1. Commit + push the Adebayo scripts to GitHub
2. Push the U-Net Adebayo notebook on account 1
3. Poll every 60 s (max 9 h)
4. Push the SegResNet Adebayo notebook on account 2 in parallel
5. Download both output sets to `results/adebayo_unet/` and `results/adebayo_segresnet/`
6. Centralise all `*_adebayo_full.csv` into `results/adebayo/`
7. Run the aggregator → `adebayo_aggregated.csv` + `adebayo_summary.csv`

### 4. Render the SSIM decay figure (Figure B / Figure 8)

```bash
python paper/figures/generate_figure8_adebayo.py \
    --adebayo_dir results/adebayo \
    --out_dir paper/figures/xai
```

## Expected runtime

| Model | n_patients | Conv3d layers | Time per Kaggle T4 |
|---|---|---|---|
| U-Net | 5 / fold × 5 folds = 25 | ≈ 40 | ≈ 5–6 h |
| SegResNet | 5 / fold × 5 folds = 25 | ≈ 50 | ≈ 6–8 h |

Both run in parallel on two separate Kaggle accounts → wall-clock ≈ 8 h.

## Expected output

`results/adebayo/{arch}_fold{N}_adebayo_full.csv` with columns

```
patient_id, organ, step, layer_randomized, ssim_vs_pristine,
in_organ_ratio, pointing_accuracy, spatial_entropy
```

For each patient × organ × step (step 0 = pristine, step k = k deepest
Conv3d layers randomised, monotonically working back to the input).

## Aggregated outputs

After `process_adebayo_results.py`:

- `adebayo_aggregated.csv` — long-form, one row per patient × organ × step × arch
- `adebayo_summary.csv` — `(arch, step) → mean ± SD of SSIM` ready for the curve
- `adebayo_quality_report.txt` — coverage QA

## Troubleshooting

**"kernel push failed: 403"** — the target Kaggle username does not match the
token in `~/.kaggle/kaggle.json`. The dual launcher swaps tokens
automatically; if you run by hand, verify with `kaggle config view`.

**"kernel run cancelled after ~12 h"** — exceeded Kaggle session limit. The
Adebayo loop has no checkpointing yet; re-launch with a reduced
`N_PATIENTS_PER_FOLD` (edit `scripts/kaggle/adebayo_kaggle.py` line 32).

**Empty SSIM column** — the SEG-GRAD-CAM step failed silently (usually CUDA
OOM on a large volume). The runner falls back to CPU but logs the patient ID.
Check the Kaggle log around the failing `PATIENT_s*` lines.
