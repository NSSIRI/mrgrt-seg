#!/usr/bin/env bash
# =============================================================================
# Adebayo cascading-randomization dual-account Kaggle launcher
# -----------------------------------------------------------------------------
# Pushes two Kaggle notebooks in parallel:
#   - account 1 (abdelhalimnssiri) → U-Net Adebayo runs (5 folds)
#   - account 2 (nssiri02)         → SegResNet Adebayo runs (5 folds)
# Polls execution, downloads CSVs, centralises in results/adebayo/.
#
# Cross-platform: tested on Linux, macOS, Windows Git Bash, WSL.
# Equivalent of scripts/kaggle/12_run_adebayo_batch.ps1 but in Bash.
#
# Prerequisites:
#   - kaggle CLI installed and on $PATH
#   - Two token files at:
#        ~/.kaggle/kaggle.json.compte1   (U-Net account)
#        ~/.kaggle/kaggle.json.compte2   (SegResNet account)
#     Each file is a backup of the original kaggle.json that this script
#     temporarily swaps in.
#   - Training notebooks <user>/<model>-fold-<N> already runned & public
#     (so they appear in the "Add Input" panel on Kaggle).
#
# Usage:
#   bash scripts/kaggle/launch_adebayo_dual.sh                 # full run
#   bash scripts/kaggle/launch_adebayo_dual.sh --unet-only     # account 1 only
#   bash scripts/kaggle/launch_adebayo_dual.sh --seg-only      # account 2 only
#   bash scripts/kaggle/launch_adebayo_dual.sh --skip-push     # don't git push
# =============================================================================
set -euo pipefail

# ----- Configuration ---------------------------------------------------------
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." &> /dev/null && pwd)"
KAGGLE_DIR="$HOME/.kaggle"
KAGGLE_JSON="$KAGGLE_DIR/kaggle.json"
TOKEN_A1="$KAGGLE_DIR/kaggle.json.compte1"
TOKEN_A2="$KAGGLE_DIR/kaggle.json.compte2"
USER_A1="${KAGGLE_USER_1:-abdelhalimnssiri}"
USER_A2="${KAGGLE_USER_2:-nssiri02}"
KAGGLE_BIN="${KAGGLE_BIN:-kaggle}"
POLL_INTERVAL=60     # seconds
MAX_RUN_MIN=540      # 9h hard timeout
RESULTS_DIR="$REPO_ROOT/results"
ADEBAYO_DIR="$RESULTS_DIR/adebayo"

mkdir -p "$ADEBAYO_DIR"

UNET_OK=0; SEG_OK=0
DO_UNET=1; DO_SEG=1; DO_PUSH=1
for arg in "$@"; do
  case "$arg" in
    --unet-only) DO_SEG=0 ;;
    --seg-only)  DO_UNET=0 ;;
    --skip-push) DO_PUSH=0 ;;
    -h|--help)
      sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "Unknown flag: $arg"; exit 1 ;;
  esac
done

# ----- Pretty print helpers --------------------------------------------------
b() { printf "\033[1;36m%s\033[0m\n" "$1"; }
g() { printf "\033[1;32m%s\033[0m\n" "$1"; }
y() { printf "\033[1;33m%s\033[0m\n" "$1"; }
r() { printf "\033[1;31m%s\033[0m\n" "$1"; }

# ----- Pre-flight checks -----------------------------------------------------
b "============================================"
b " Adebayo cascading randomisation — dual run "
b "============================================"

command -v "$KAGGLE_BIN" >/dev/null 2>&1 || {
  r "[ERR] kaggle CLI not on PATH (set KAGGLE_BIN env var if needed)"; exit 1;
}
[[ -f "$TOKEN_A1" ]] || {
  r "[ERR] $TOKEN_A1 missing — backup compte 1: cp $KAGGLE_JSON $TOKEN_A1"; exit 1;
}
if [[ $DO_SEG -eq 1 ]]; then
  if [[ ! -f "$TOKEN_A2" ]]; then
    y "[WARN] $TOKEN_A2 missing — continuing with account 1 only"
    DO_SEG=0
  fi
fi

cd "$REPO_ROOT"

# ----- 0. Git push -----------------------------------------------------------
if [[ $DO_PUSH -eq 1 ]]; then
  b "\n[0] git status / push"
  git add scripts/run_adebayo_analysis.py scripts/kaggle/adebayo_kaggle.py \
          scripts/kaggle/make_adebayo_notebook.py \
          scripts/process_adebayo_results.py src/xai/ 2>/dev/null || true
  if ! git diff --cached --quiet 2>/dev/null; then
    git commit -m "Adebayo: scripts + dual-account launcher" || true
  fi
  git push || y "  (push failed or nothing to push)"
fi

# ----- Helper: push + poll one model run ------------------------------------
run_one() {
  local model="$1" user="$2" token="$3"

  b "\n============================================"
  b " ADEBAYO ${model^^} batch ($user)"
  b "============================================"

  cp "$token" "$KAGGLE_JSON"
  chmod 600 "$KAGGLE_JSON"

  local kernel="${user}/mrgrt-adebayo-${model}-batch"
  local tmpdir
  tmpdir="$(mktemp -d -t kaggle_adebayo_${model}_XXXXXX)"

  echo "Generating notebook in $tmpdir"
  python3 "$REPO_ROOT/scripts/kaggle/make_adebayo_notebook.py" \
      --model "$model" --username "$user" --out_dir "$tmpdir" \
    || { r "[ERR] notebook generation failed"; rm -rf "$tmpdir"; return 1; }

  echo "Pushing kernel..."
  ( cd "$tmpdir" && "$KAGGLE_BIN" kernels push -p . ) \
    || { r "[ERR] kaggle kernels push failed"; rm -rf "$tmpdir"; return 1; }

  echo "Polling every ${POLL_INTERVAL}s (max ${MAX_RUN_MIN} min)..."
  local start_ts
  start_ts=$(date +%s)
  while :; do
    sleep "$POLL_INTERVAL"
    local elapsed_min
    elapsed_min=$(( ($(date +%s) - start_ts) / 60 ))
    local status
    status=$("$KAGGLE_BIN" kernels status "$kernel" 2>&1 || true)
    echo "  [${elapsed_min} min] $status"
    case "$status" in
      *complete*|*COMPLETE*) break ;;
      *error*|*ERROR*|*cancel*|*CANCEL*|*fail*|*FAIL*)
        r "[ERR] kernel run failed: $status"; rm -rf "$tmpdir"; return 1 ;;
    esac
    if (( elapsed_min > MAX_RUN_MIN )); then
      r "[TIMEOUT] ${MAX_RUN_MIN} min exceeded"; rm -rf "$tmpdir"; return 1
    fi
  done

  echo "Downloading outputs..."
  local outdir="$RESULTS_DIR/adebayo_${model}"
  rm -rf "$outdir"; mkdir -p "$outdir"
  "$KAGGLE_BIN" kernels output "$kernel" -p "$outdir" >/dev/null 2>&1 || true
  echo "Outputs → $outdir"

  rm -rf "$tmpdir"
  return 0
}

# ----- Run U-Net on account 1 ------------------------------------------------
if [[ $DO_UNET -eq 1 ]]; then
  if run_one "unet" "$USER_A1" "$TOKEN_A1"; then
    UNET_OK=1
    g "U-Net Adebayo OK"
  else
    r "U-Net Adebayo FAILED"
  fi
fi

# ----- Run SegResNet on account 2 --------------------------------------------
if [[ $DO_SEG -eq 1 ]]; then
  if run_one "segresnet" "$USER_A2" "$TOKEN_A2"; then
    SEG_OK=1
    g "SegResNet Adebayo OK"
  else
    r "SegResNet Adebayo FAILED"
  fi
fi

# ----- Restore account 1 token -----------------------------------------------
cp "$TOKEN_A1" "$KAGGLE_JSON"
chmod 600 "$KAGGLE_JSON"
g "\naccess_token restored to compte 1"

# ----- Centralise CSVs -------------------------------------------------------
b "\nCentralising CSVs into $ADEBAYO_DIR"
find "$RESULTS_DIR" -name "*_adebayo_full.csv" -type f 2>/dev/null \
  | while IFS= read -r f; do
      cp "$f" "$ADEBAYO_DIR/"
      echo "  $(basename "$f") → $ADEBAYO_DIR/"
    done

# ----- Aggregate + report ----------------------------------------------------
b "\nAggregating with process_adebayo_results.py"
python3 "$REPO_ROOT/scripts/process_adebayo_results.py" \
    --adebayo_dir "$ADEBAYO_DIR" \
  || y "  (aggregation step skipped — no CSVs yet?)"

# ----- Final summary ---------------------------------------------------------
b "\n============================================"
b " ADEBAYO BATCH FINISHED"
b "============================================"
printf "U-Net      : %s\n" "$( ((UNET_OK)) && echo OK || echo SKIP/FAIL)"
printf "SegResNet  : %s\n" "$( ((SEG_OK))  && echo OK || echo SKIP/FAIL)"
echo
echo "CSV files produced:"
ls -1 "$ADEBAYO_DIR"/*.csv 2>/dev/null | sed 's/^/  /'

echo
b "Next step — render Figure B (SSIM decay curves):"
echo "  python paper/figures/generate_figure8_adebayo.py \\"
echo "      --adebayo_dir $ADEBAYO_DIR \\"
echo "      --out_dir paper/figures/xai"
