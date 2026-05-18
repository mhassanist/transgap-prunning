#!/bin/bash
# =============================================================================
# truba_revision_orthog.slurm
# Items 7 + 12 — Orthogonality validation, ~2h
# SUBMIT: cd /arf/scratch/mhassan/transgap_v2 && sbatch /arf/home/mhassan/transgap_v2/scripts/truba_revision_orthog.slurm [backbone]
# backbone: r56_cifar10 | r110_cifar10 | r50_imagenet | all  (default: all)
# =============================================================================
#SBATCH -J tg2_rev_orthog
#SBATCH -A mhassan
#SBATCH -p akya-cuda
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 10
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/revision_orthog_%j.out
#SBATCH --error=logs/revision_orthog_%j.err

BACKBONE=${1:-all}

echo "============================================================"
echo "Job: $SLURM_JOB_ID on $SLURM_NODELIST"
echo "Items 7+12: Orthogonality validation | backbone=$BACKBONE"
echo "Start: $(date)"
echo "============================================================"

SCRATCH=/arf/scratch/$USER/transgap_v2
CODE=/arf/home/$USER/transgap_v2

mkdir -p $SCRATCH/logs $SCRATCH/checkpoints
cd $SCRATCH

module purge
module load comp/python/miniconda3

python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

cd $CODE

python -m experiments.run_orthogonality \
    --backbone $BACKBONE \
    --checkpoint_dir $SCRATCH/checkpoints \
    --data_path $SCRATCH/data \
    --imagenet_path /arf/scratch/$USER/imagenet \
    --save_dir $SCRATCH/checkpoints

echo "Completed: $(date)"
exit
