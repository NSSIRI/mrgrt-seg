#!/bin/bash
# =====================================================================
# Entraine 1 modele x 1 fold sur MARWAN - partition gpu-prodq (7 jours)
#
# Usage (depuis ~/mrgrt-seg sur MARWAN) :
#   sbatch scripts/slurm/train_one.sbatch unet 0
#   sbatch scripts/slurm/train_one.sbatch segresnet 2
#   sbatch scripts/slurm/train_one.sbatch unet 0 configs/ct.yaml
# =====================================================================
#SBATCH --job-name=mrgrt_train
#SBATCH --partition=gpu-prodq
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/train_%x_%j.out
#SBATCH --error=logs/train_%x_%j.err

set -euo pipefail
mkdir -p logs

MODEL=${1:?"usage: sbatch train_one.sbatch <unet|segresnet> <fold> [config]"}
FOLD=${2:?"usage: sbatch train_one.sbatch <unet|segresnet> <fold> [config]"}
CONFIG=${3:-configs/default.yaml}

echo "============================================"
echo " Job ID  : $SLURM_JOB_ID"
echo " Node    : $SLURMD_NODENAME"
echo " Modele  : $MODEL  | Fold : $FOLD"
echo " Config  : $CONFIG"
echo " Started : $(date)"
echo "============================================"

module purge
module load Anaconda3/2024.02-1
module load CUDA/11.4.1
source activate mrgrt-seg
export PYTHONNOUSERSITE=1

python -c "import torch; assert torch.cuda.is_available(), 'CUDA indisponible sur ce noeud!'"

python scripts/train.py --model "$MODEL" --fold "$FOLD" --config "$CONFIG"

echo "Finished : $(date)"
