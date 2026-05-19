#!/bin/bash
# =====================================================================
# Entraine 1 modele x 1 fold sur MARWAN - partition gpu-prodq
#
# Usage (depuis ~/mrgrt-seg sur MARWAN) :
#   sbatch scripts/slurm/train_one.sbatch unet 0
#   sbatch scripts/slurm/train_one.sbatch segresnet 2
#   sbatch scripts/slurm/train_one.sbatch unet 0 configs/ct.yaml
#   sbatch scripts/slurm/train_one.sbatch unet 0 configs/default.yaml --epochs 5
#   sbatch scripts/slurm/train_one.sbatch unet 0 configs/default.yaml --resume auto
# =====================================================================
#SBATCH --job-name=mrgrt_train
#SBATCH --partition=gpu-prodq
#SBATCH --qos=gpu
#SBATCH --account=gpu_users
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

MODEL=${1:?"usage: sbatch train_one.sbatch <unet|segresnet> <fold> [config] [extra args...]"}
FOLD=${2:?"usage: sbatch train_one.sbatch <unet|segresnet> <fold> [config] [extra args...]"}
CONFIG=${3:-configs/default.yaml}
# Args supplementaires (ex: --epochs 5 --resume) transmis tels quels a train.py
EXTRA_ARGS=("${@:4}")

echo "============================================"
echo " Job ID  : $SLURM_JOB_ID"
echo " Node    : $SLURMD_NODENAME"
echo " Modele  : $MODEL  | Fold : $FOLD"
echo " Config  : $CONFIG"
if [ ${#EXTRA_ARGS[@]} -gt 0 ]; then
    echo " Extra   : ${EXTRA_ARGS[*]}"
fi
echo " Started : $(date)"
echo "============================================"

module purge
module load Anaconda3/2024.02-1
module load CUDA/11.4.1
source activate mrgrt-seg
export PYTHONNOUSERSITE=1

python -c "import torch; assert torch.cuda.is_available(), 'CUDA indisponible sur ce noeud!'"

python scripts/train.py --model "$MODEL" --fold "$FOLD" --config "$CONFIG" "${EXTRA_ARGS[@]}"

echo "Finished : $(date)"
