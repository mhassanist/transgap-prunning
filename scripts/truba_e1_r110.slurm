#!/bin/bash
#SBATCH --job-name=tg_e1_r110
#SBATCH --account=mhassan
#SBATCH --partition=akya-cuda
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=0-04:00:00
#SBATCH --output=/arf/scratch/mhassan/transgap_v2/logs/e1_r110_%j_%x.out
#SBATCH --error=/arf/scratch/mhassan/transgap_v2/logs/e1_r110_%j_%x.err
#SBATCH --chdir=/arf/scratch/mhassan/transgap_v2

# E1: Cross-Backbone Correlation — single block fine-tune
#
# Usage:
#   sbatch scripts/truba_e1_r110.slurm <block_name>
#   e.g.  sbatch scripts/truba_e1_r110.slurm layer2.12
#
# Time budget: 150 epochs on CIFAR-10 ≈ 1.5–2 h on akya-cuda  →  4 h wall time

BLOCK=$1
if [ -z "$BLOCK" ]; then
    echo "ERROR: block name required as first argument"
    echo "Usage: sbatch $0 <block_name>"
    exit 1
fi

SCRATCH=/arf/scratch/mhassan/transgap_v2
CODE=/arf/home/mhassan/transgap_v2

mkdir -p $SCRATCH/{logs,checkpoints}

echo "============================================================"
echo "Job   : $SLURM_JOB_ID  on  $(hostname)"
echo "Task  : E1 single-block fine-tune"
echo "Block : $BLOCK"
echo "Start : $(date)"
echo "============================================================"

module purge
module load comp/python/miniconda3
source activate

python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
nvidia-smi --query-gpu=name,memory.total --format=csv 2>/dev/null

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

cd $CODE

python -m experiments.run_e1_r110_correlation \
    --block     "$BLOCK" \
    --checkpoint $SCRATCH/checkpoints/resnet110_cifar10_baseline.pth \
    --data_path  $SCRATCH/data \
    --save_dir   $SCRATCH/checkpoints

echo "============================================================"
echo "Done  : $(date)"
echo "============================================================"
