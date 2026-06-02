#!/usr/bin/env bash
# Local smoke test for the Adebayo runner — 1 patient, 1 organ, 1 fold.
# Useful to validate the pipeline end-to-end before paying for the 5-8h Kaggle run.
#
# Usage:
#   bash scripts/kaggle/smoke_test_adebayo.sh unet 0 /path/to/best.pt
#   bash scripts/kaggle/smoke_test_adebayo.sh segresnet 0 /path/to/best.pt
set -euo pipefail

MODEL="${1:?'usage: smoke_test_adebayo.sh <unet|segresnet> <fold> <ckpt>'}"
FOLD="${2:?'fold required'}"
CKPT="${3:?'checkpoint path required'}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." &> /dev/null && pwd)"
OUT_DIR="$REPO_ROOT/results/adebayo_smoke"
mkdir -p "$OUT_DIR"

cd "$REPO_ROOT"

python3 scripts/run_adebayo_analysis.py \
    --model "$MODEL" \
    --fold "$FOLD" \
    --ckpt "$CKPT" \
    --config configs/default.yaml \
    --n_patients 1 \
    --target_organs 4 \
    --out_dir "$OUT_DIR"

echo
echo "Smoke test done. Outputs:"
ls -lh "$OUT_DIR"
echo
echo "Sanity check the CSV layout:"
head -3 "$OUT_DIR/${MODEL}_fold${FOLD}_adebayo_full.csv"
