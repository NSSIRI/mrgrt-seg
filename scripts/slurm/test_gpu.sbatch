#!/bin/bash
# =====================================================================
# Test GPU rapide sur MARWAN - valide que l'entrainement GPU fonctionne
# Partition gpu-testq (limite 1h) - parfait pour un test
#
# Usage (depuis ~/mrgrt-seg sur MARWAN) :
#   sbatch scripts/slurm/test_gpu.sbatch
# =====================================================================
#SBATCH --job-name=mrgrt_gputest
#SBATCH --partition=gpu-testq
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/gputest_%j.out
#SBATCH --error=logs/gputest_%j.err

set -euo pipefail
mkdir -p logs

echo "============================================"
echo " Job ID  : $SLURM_JOB_ID"
echo " Node    : $SLURMD_NODENAME"
echo " Started : $(date)"
echo "============================================"

# --- Environnement ---
module purge
module load Anaconda3/2024.02-1
module load CUDA/11.4.1
source activate mrgrt-seg
export PYTHONNOUSERSITE=1

# --- Verification GPU ---
echo ""
echo "--- nvidia-smi ---"
nvidia-smi
echo ""
echo "--- PyTorch CUDA ---"
python -c "import torch; print('CUDA dispo :', torch.cuda.is_available()); print('GPU :', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'AUCUN'); print('VRAM Go :', round(torch.cuda.get_device_properties(0).total_memory/1e9,1) if torch.cuda.is_available() else 0)"

# --- Smoke test complet (cette fois sur GPU) ---
echo ""
echo "--- Smoke test ---"
python smoke_test.py

echo ""
echo "Finished : $(date)"
