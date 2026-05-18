#!/bin/bash
# =====================================================================
# 01_marwan_setup.sh — À exécuter SUR MARWAN, une seule fois.
#
# Prépare la destination des données :
#   - crée $SCRATCH/mrgrt-seg/data/
#   - crée le symlink ~/mrgrt-seg/data -> $SCRATCH/mrgrt-seg/data
#   - affiche le chemin résolu de $SCRATCH (à reporter dans 02_local_rsync.sh)
#
# Usage :
#   cd ~/mrgrt-seg
#   bash scripts/transfer/01_marwan_setup.sh
# =====================================================================
set -euo pipefail

PROJECT_DIR="$HOME/mrgrt-seg"

echo "=== 01_marwan_setup ==="
echo ""

# 1) Vérifier $SCRATCH
if [ -z "${SCRATCH:-}" ]; then
    echo "ERREUR : la variable \$SCRATCH n'est pas définie."
    echo ""
    echo "Sur MARWAN, exécute d'abord :"
    echo "  echo \$SCRATCH"
    echo ""
    echo "Si la variable n'existe pas, demande à l'admin MARWAN le chemin du"
    echo "scratch (ou utilise le chemin équivalent : /scratch/\$USER, /work/\$USER, etc.)"
    echo "puis :  export SCRATCH=/le/chemin/donné  et relance ce script."
    exit 1
fi
echo "SCRATCH résolu : $SCRATCH"

# 2) Vérifier que SCRATCH est accessible en écriture
if [ ! -w "$SCRATCH" ]; then
    echo "ERREUR : \$SCRATCH ($SCRATCH) n'est pas accessible en écriture."
    exit 1
fi

# 3) Créer la destination data
DATA_DEST="$SCRATCH/mrgrt-seg/data"
mkdir -p "$DATA_DEST"
echo "Dossier créé : $DATA_DEST"

# 4) Créer le symlink dans le projet
cd "$PROJECT_DIR"
if [ -L "data" ]; then
    echo "Le symlink data/ existe déjà — on le re-pointe vers $DATA_DEST"
    rm "data"
elif [ -d "data" ]; then
    # data/ existe comme vrai dossier (probablement vide après git clone si .gitkeep)
    if [ -z "$(ls -A data 2>/dev/null)" ]; then
        echo "Dossier data/ vide — on le supprime pour créer le symlink"
        rmdir "data"
    else
        echo "ATTENTION : data/ existe et N'EST PAS vide. Contenu :"
        ls -la data/ | head -10
        echo ""
        read -p "Le supprimer pour créer le symlink ? [y/N] " ans
        if [ "$ans" = "y" ] || [ "$ans" = "Y" ]; then
            rm -rf "data"
        else
            echo "Abandon. Renomme ou déplace data/ manuellement puis relance."
            exit 1
        fi
    fi
fi
ln -s "$DATA_DEST" data
echo "Symlink créé :"
ls -la data | head -1

# 5) Récap pour l'utilisateur
echo ""
echo "============================================================"
echo " À reporter dans 02_local_rsync.sh (sur ton PC, WSL2) :"
echo ""
echo "   export MARWAN_SCRATCH=\"$SCRATCH\""
echo ""
echo " Le dataset sera transféré dans :"
echo "   ${SCRATCH}/mrgrt-seg/data/"
echo ""
echo " Et accessible via :"
echo "   ${PROJECT_DIR}/data  (symlink)"
echo "============================================================"
