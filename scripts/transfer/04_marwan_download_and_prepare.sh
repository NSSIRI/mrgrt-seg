#!/bin/bash
# =====================================================================
# 04_marwan_download_and_prepare.sh
#
# Telecharge TotalSegmentator MRI v2.0.0 depuis Zenodo (5.1 GB) puis le
# convertit en NIfTI multi-classes (5 classes : bg + 4 OAR thoraciques)
# directement sur MARWAN. Plus rapide que rsync depuis le PC : MARWAN
# telecharge a la vitesse du datacenter.
#
# Prerequis :
#   - 01_marwan_setup.sh deja execute (symlink ~/mrgrt-seg/data -> $SCRATCH)
#   - $SCRATCH defini : export SCRATCH=/scratch/users/$USER
#   - Env conda mrgrt-seg active : source activate mrgrt-seg
#
# Usage :
#   cd ~/mrgrt-seg
#   export SCRATCH=/scratch/users/$USER     # si pas deja
#   source activate mrgrt-seg               # active l'env python
#   bash scripts/transfer/04_marwan_download_and_prepare.sh
#
# Le script est idempotent : si le zip est deja la, il ne re-download pas ;
# si l'extraction est faite, il ne re-extrait pas. Tu peux relancer sans risque.
# =====================================================================
set -euo pipefail

# --- Configuration ---
ZENODO_URL="https://zenodo.org/records/14710732/files/TotalsegmentatorMRI_dataset_v200.zip?download=1"
EXPECTED_MD5="54638f4cb883ce3b34225195358c398f"

# --- Resolution des chemins ---
if [ -z "${SCRATCH:-}" ]; then
    echo "ERREUR : \$SCRATCH non defini."
    echo "Faire :  export SCRATCH=/scratch/users/\$USER"
    exit 1
fi

DL_DIR="$SCRATCH/mrgrt-seg/downloads"
ZIP="$DL_DIR/TotalsegmentatorMRI_dataset_v200.zip"
RAW_DIR="$DL_DIR/raw_totalseg"
DATA_DST="$SCRATCH/mrgrt-seg/data"
REPO_DIR="$HOME/mrgrt-seg"

echo "============================================================"
echo " MARWAN download + prepare TotalSegmentator MRI v2.0.0"
echo "============================================================"
echo " SCRATCH  : $SCRATCH"
echo " ZIP      : $ZIP"
echo " RAW      : $RAW_DIR"
echo " DATA     : $DATA_DST"
echo ""

mkdir -p "$DL_DIR"

# --- [1/4] Telechargement ---
echo "=== [1/4] Telechargement du zip (5.1 GB) ==="
if [ -f "$ZIP" ]; then
    SIZE_MB=$(du -m "$ZIP" | awk '{print $1}')
    echo "Zip deja present (${SIZE_MB} MB). Skip download."
    echo "(Pour forcer un re-download :  rm $ZIP  et relance)"
else
    echo "Lancement wget (~5-15 min selon la liaison MARWAN)..."
    # --continue : reprend un download partiel
    # --tries=5 : tolerance aux micro-coupures
    wget --continue --tries=5 -O "$ZIP" "$ZENODO_URL"
fi

# --- [2/4] Verification MD5 ---
echo ""
echo "=== [2/4] Verification MD5 ==="
ACTUAL_MD5=$(md5sum "$ZIP" | awk '{print $1}')
if [ "$ACTUAL_MD5" = "$EXPECTED_MD5" ]; then
    echo "MD5 OK : $ACTUAL_MD5"
else
    echo "ERREUR : MD5 ne correspond pas !"
    echo "  attendu : $EXPECTED_MD5"
    echo "  obtenu  : $ACTUAL_MD5"
    echo "Le download est probablement corrompu."
    echo "Faire :  rm $ZIP  et relance."
    exit 1
fi

# --- [3/4] Decompression ---
echo ""
echo "=== [3/4] Decompression ==="
if [ -d "$RAW_DIR" ] && [ -n "$(ls -A "$RAW_DIR" 2>/dev/null)" ]; then
    N=$(find "$RAW_DIR" -maxdepth 2 -type d -name "s[0-9]*" | wc -l)
    echo "Dossier $RAW_DIR contient deja $N sous-dossiers patient. Skip unzip."
    echo "(Pour forcer :  rm -rf $RAW_DIR  et relance)"
else
    mkdir -p "$RAW_DIR"
    echo "Extraction en cours (peut prendre 2-5 min)..."
    unzip -q "$ZIP" -d "$RAW_DIR"
    N=$(find "$RAW_DIR" -maxdepth 2 -type d -name "s[0-9]*" | wc -l)
    echo "Extraction OK : $N sous-dossiers patient trouves."
fi

# --- [4/4] Conversion vers NIfTI multi-classes ---
echo ""
echo "=== [4/4] Conversion vers le format projet ==="
if ! command -v python &> /dev/null; then
    echo "ERREUR : python non trouve dans le PATH."
    echo "Activer l'env conda :  source activate mrgrt-seg  puis relance."
    exit 1
fi

# Empecher d'ecraser une data/ deja peuplee sans prevenir
if [ -d "$DATA_DST" ] && [ "$(find "$DATA_DST" -maxdepth 1 -type d -name 'PATIENT_*' | wc -l)" -gt 0 ]; then
    EXISTING=$(find "$DATA_DST" -maxdepth 1 -type d -name 'PATIENT_*' | wc -l)
    echo "ATTENTION : $DATA_DST contient deja $EXISTING patients."
    read -p "Re-executer la conversion (ecrasement) ? [y/N] " ans
    if [ "$ans" != "y" ] && [ "$ans" != "Y" ]; then
        echo "Abandon. (data/ conservee telle quelle.)"
        exit 0
    fi
fi

cd "$REPO_DIR"
python scripts/prepare_totalsegmentator.py \
    --src "$RAW_DIR" \
    --dst "$DATA_DST" \
    --min_classes 2 \
    --require_lungs

# --- Recap final ---
echo ""
echo "============================================================"
N_FINAL=$(find "$DATA_DST" -maxdepth 1 -type d -name 'PATIENT_*' | wc -l)
echo " Termine. $N_FINAL patients dans $DATA_DST"
echo ""
echo " Verification d'integrite :"
echo "   python scripts/transfer/03_marwan_verify.py"
echo ""
echo " Si tout est OK, prochaine etape : sanity-run sur 1 fold :"
echo "   sbatch scripts/slurm/train_one.sbatch unet 0"
echo "============================================================"
