#!/bin/bash
# =====================================================================
# Cree une archive tar.gz du dataset clean (303 patients) sur MARWAN
# pour permettre un download rapide vers le PC.
#
# Usage (depuis ~/mrgrt-seg sur MARWAN, env conda actif) :
#   bash scripts/transfer/06_marwan_create_archive.sh
# =====================================================================
set -euo pipefail

echo "=== 06_marwan_create_archive ==="

# --- Resoudre SCRATCH ------------------------------------------------
if [ -z "${SCRATCH:-}" ]; then
    if [ -d "/scratch/users/$USER" ]; then
        export SCRATCH="/scratch/users/$USER"
    else
        echo "ERREUR : SCRATCH non defini et /scratch/users/$USER absent." >&2
        echo "  -> export SCRATCH=/le/chemin && relance" >&2
        exit 1
    fi
fi

SRC="$SCRATCH/mrgrt-seg/data_thorax_complet"
DST="$SCRATCH/mrgrt-seg/downloads/data_thorax_complet.tar.gz"

if [ ! -d "$SRC" ]; then
    echo "ERREUR : $SRC absent." >&2
    echo "  Lance d'abord scripts/transfer/05_filter_quality.py (sans --dry-run)" >&2
    exit 1
fi

# --- Compter les patients --------------------------------------------
N_PAT=$(ls -d "$SRC"/PATIENT_* 2>/dev/null | wc -l)
echo "Patients trouves : $N_PAT"
SIZE_RAW=$(du -sh "$SRC" | awk '{print $1}')
echo "Taille brute    : $SIZE_RAW"

if [ "$N_PAT" -lt 100 ]; then
    echo "ATTENTION : seulement $N_PAT patients, c'est anormalement peu." >&2
    read -p "Continuer quand meme ? (o/N) " ans
    [ "$ans" = "o" ] || [ "$ans" = "O" ] || exit 1
fi

# --- Creer l'archive (NIfTI deja gz : on saute la compression) -------
mkdir -p "$(dirname "$DST")"

if [ -f "$DST" ]; then
    echo "Archive existante : $DST"
    read -p "Ecraser ? (o/N) " ans
    if [ "$ans" != "o" ] && [ "$ans" != "O" ]; then
        echo "Annule." ; exit 0
    fi
    rm "$DST"
fi

echo ""
echo "Creation de l'archive (NIfTI deja .gz, donc tar sans recompression)..."
# Les fichiers etant deja .nii.gz, -z ferait juste perdre du temps CPU
# Mais l'extension .tar.gz est gardee pour la convention. On utilise -I 'gzip -1'
# pour avoir un peu de compression sur les noms de dossiers + structure.
cd "$(dirname "$SRC")"
SRC_BASENAME=$(basename "$SRC")

time tar -cf "$DST" "$SRC_BASENAME"

SIZE_TAR=$(du -h "$DST" | awk '{print $1}')
echo ""
echo "============================================"
echo " Archive cree avec succes."
echo "============================================"
echo " Chemin    : $DST"
echo " Taille    : $SIZE_TAR"
echo " Patients  : $N_PAT"
echo ""
echo " Commande a lancer sur ton PC (PowerShell ou Git Bash) :"
echo ""
echo "   scp a.nssiri@hpc-login.marwan.ma:$DST C:\\Users\\Lenovo\\Desktop\\mrgrt_seg\\"
echo ""
echo " Apres download, sur PC (Git Bash ou WSL) pour extraire :"
echo "   cd C:\\Users\\Lenovo\\Desktop\\mrgrt_seg"
echo "   tar -xf data_thorax_complet.tar.gz"
echo ""
echo " Ou sur Windows 10+ direct dans Explorer : clic droit -> Extraire tout."
echo "============================================"
