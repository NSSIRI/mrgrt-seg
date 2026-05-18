#!/bin/bash
# =====================================================================
# 02_local_rsync.sh — À exécuter SUR TON PC, depuis WSL2 Ubuntu.
#
# Transfère le dossier data/ (250 patients, ~1.85 GB) vers MARWAN via rsync.
# Resumable : Ctrl+C puis relance reprend où ça s'était arrêté.
#
# Pré-requis :
#   1) L'étape 01_marwan_setup.sh a été exécutée sur MARWAN.
#   2) SSH MARWAN fonctionne sans mot de passe (clé) — sinon mot de passe
#      demandé à chaque connexion rsync.
#
# Configuration : exporter les 3 variables AVANT de lancer :
#   export MARWAN_USER="ton_login"
#   export MARWAN_HOST="hpc.marwan.ma"
#   export MARWAN_SCRATCH="/scratch/ton_login"    # affiché par l'étape 1
#
# Usage :
#   bash /mnt/c/Users/Lenovo/Desktop/mrgrt_seg/scripts/transfer/02_local_rsync.sh
# =====================================================================
set -euo pipefail

# --------- Configuration ---------
LOCAL_DATA="${LOCAL_DATA:-/mnt/c/Users/Lenovo/Desktop/mrgrt_seg/data}"
MARWAN_USER="${MARWAN_USER:-}"
MARWAN_HOST="${MARWAN_HOST:-}"
MARWAN_SCRATCH="${MARWAN_SCRATCH:-}"
# ----------------------------------

# Vérifications
if [ -z "$MARWAN_USER" ] || [ -z "$MARWAN_HOST" ] || [ -z "$MARWAN_SCRATCH" ]; then
    echo "ERREUR : variables manquantes. Exporter avant de lancer :"
    echo "  export MARWAN_USER=\"ton_login\""
    echo "  export MARWAN_HOST=\"hpc.marwan.ma\""
    echo "  export MARWAN_SCRATCH=\"/scratch/ton_login\""
    exit 1
fi

if [ ! -d "$LOCAL_DATA" ]; then
    echo "ERREUR : dossier local introuvable : $LOCAL_DATA"
    echo "Ajuste LOCAL_DATA si le projet est ailleurs."
    exit 1
fi

REMOTE_DATA="${MARWAN_USER}@${MARWAN_HOST}:${MARWAN_SCRATCH}/mrgrt-seg/data/"

echo "============================================================"
echo " Transfert MRgRT — dataset vers MARWAN"
echo "============================================================"
echo " Source : $LOCAL_DATA"
echo " Cible  : $REMOTE_DATA"
echo ""

# Comptage local (sanity)
LOCAL_COUNT=$(find "$LOCAL_DATA" -maxdepth 1 -type d -name "PATIENT_s*" | wc -l)
LOCAL_SIZE=$(du -sh "$LOCAL_DATA" | awk '{print $1}')
echo " Local  : $LOCAL_COUNT patients, taille totale $LOCAL_SIZE"
echo ""

# Test SSH rapide
echo "Test SSH..."
if ! ssh -o ConnectTimeout=10 -o BatchMode=yes "${MARWAN_USER}@${MARWAN_HOST}" "echo OK" 2>/dev/null; then
    echo "  ssh sans mot de passe NON configuré (clé absente / mauvaise / verrouillée)."
    echo "  Le transfert va te demander un mot de passe à chaque fichier — pénible."
    echo "  Recommandation : configurer ssh-keygen + ssh-copy-id avant."
    echo ""
    read -p "Continuer quand même ? [y/N] " ans
    if [ "$ans" != "y" ] && [ "$ans" != "Y" ]; then exit 1; fi
fi

# Vérifier que la destination existe (étape 1 faite ?)
echo "Vérification de la destination distante..."
if ! ssh "${MARWAN_USER}@${MARWAN_HOST}" "test -d '${MARWAN_SCRATCH}/mrgrt-seg/data'"; then
    echo "ERREUR : ${MARWAN_SCRATCH}/mrgrt-seg/data n'existe pas sur MARWAN."
    echo "Exécute d'abord 01_marwan_setup.sh sur MARWAN."
    exit 1
fi

echo ""
echo "=== Lancement rsync ==="
echo "(Ctrl+C interrompt proprement, relance reprend où ça s'est arrêté.)"
echo ""

# rsync -a archive (préserve), -v verbeux, -P progress + partial (reprise),
# -K garde les symlinks de répertoire côté distant (évite de casser le symlink data/)
# --human-readable : tailles lisibles
# --stats : récap final
rsync -avPK --human-readable --stats \
    "${LOCAL_DATA}/" "${REMOTE_DATA}"

echo ""
echo "============================================================"
echo " Transfert terminé."
echo " Vérifier maintenant côté MARWAN :"
echo "   ssh ${MARWAN_USER}@${MARWAN_HOST}"
echo "   cd ~/mrgrt-seg && source activate mrgrt-seg"
echo "   python scripts/transfer/03_marwan_verify.py"
echo "============================================================"
