#!/bin/bash
# =====================================================================
# Array job - lance les 5 folds en parallele pour un modele donne
# Partition gpu-prodq (7 jours). SLURM distribue les 5 taches sur les
# noeuds GPU disponibles (gpu51, gpu52).
#
# Usage (depuis ~/mrgrt-seg sur MARWAN) :
#   sbatch --export=MODEL=unet      scripts/slurm/train_5fold_array.sbatch
#   sbatch --export=MODEL=segresnet scripts/slurm/train_5fold_array.sbatch
#   sbatch --export=MODEL=unet,CONFIG=configs/ct.yaml scripts/slurm/train_5fold_array.sbatch
# =====================================================================
#SBATCH --job-name=mrgrt_5fold
#SBATCH --partition=gpu-prodq
#SBATCH --gres=gpu:1
#SBATCH --array=0-4
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/5fold_%x_%A_%a.out
#SBATCH --error=logs/5fold_%x_%A_%a.err

set -euo pipefail
mkdir -p logs

MODEL=${MODEL:-unet}
CONFIG=${CONFIG:-configs/default.yaml}
FOLD=$SLURM_ARRAY_TASK_ID

echo "============================================"
echo " Array task : $SLURM_ARRAY_TASK_ID / 4"
echo " Job ID     : $SLURM_JOB_ID"
echo " Node       : $SLURMD_NODENAME"
echo " Modele     : $MODEL  | Fold : $FOLD"
echo " Started    : $(date)"
echo "============================================"

module purge
module load Anaconda3/2024.02-1
module load CUDA/11.4.1
source activate mrgrt-seg
export PYTHONNOUSERSITE=1

python -c "import torch; assert torch.cuda.is_available(), 'CUDA indisponible!'"

python scripts/train.py --model "$MODEL" --fold "$FOLD" --config "$CONFIG"

echo "Finished : $(date)"
