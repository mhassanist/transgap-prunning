#!/bin/bash
# =============================================================================
# truba_revision_multiseed.slurm
# Item 6 — Multi-seed runs, ~12h per setting per seed
# SUBMIT: cd /arf/scratch/mhassan/transgap_v2 && sbatch /arf/home/mhassan/transgap_v2/scripts/truba_revision_multiseed.slurm <setting> <seed>
#
# Settings:  r56_cifar10_f05 | r56_cifar10_f06 | r110_cifar10_f06
#            r50_imagenet_f06 | r50_imagenet_f05
# Seeds:     42 | 123 | 456
#
# Example:
#   sbatch truba_revision_multiseed.slurm r56_cifar10_f05 42
#   sbatch truba_revision_multiseed.slurm r50_imagenet_f06 123
# =============================================================================
#SBATCH -J tg2_rev_seed
#SBATCH -A mhassan
#SBATCH -p akya-cuda
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 10
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/revision_seed_%j.out
#SBATCH --error=logs/revision_seed_%j.err

SETTING=${1:-r56_cifar10_f05}
SEED=${2:-42}

echo "============================================================"
echo "Job: $SLURM_JOB_ID on $SLURM_NODELIST"
echo "Item 6: Multi-seed | setting=$SETTING seed=$SEED"
echo "Start: $(date)"
echo "============================================================"

SCRATCH=/arf/scratch/$USER/transgap_v2
CODE=/arf/home/$USER/transgap_v2

mkdir -p $SCRATCH/logs $SCRATCH/checkpoints
cd $SCRATCH

module purge
module load comp/python/miniconda3

python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
nvidia-smi --query-gpu=name,memory.total --format=csv 2>/dev/null
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

cd $CODE

python -m experiments.run_multiseed \
    --setting $SETTING \
    --seed $SEED \
    --checkpoint_dir $SCRATCH/checkpoints \
    --data_path $SCRATCH/data \
    --imagenet_path /arf/scratch/$USER/imagenet \
    --save_dir $SCRATCH/checkpoints

echo "Completed: $(date)"
exit
