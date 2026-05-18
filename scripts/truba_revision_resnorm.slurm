#!/bin/bash
# =============================================================================
# truba_revision_resnorm.slurm
# Item 8 — TG vs ||F(X)||/||X|| ablation, ~16h (3 criteria × 150 epochs)
# SUBMIT: cd /arf/scratch/mhassan/transgap_v2 && sbatch /arf/home/mhassan/transgap_v2/scripts/truba_revision_resnorm.slurm [seed]
# seed: 42 | 123 | 456  (default: 42)
# =============================================================================
#SBATCH -J tg2_rev_resnorm
#SBATCH -A mhassan
#SBATCH -p akya-cuda
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 10
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/revision_resnorm_%j.out
#SBATCH --error=logs/revision_resnorm_%j.err

SEED=${1:-42}

echo "============================================================"
echo "Job: $SLURM_JOB_ID on $SLURM_NODELIST"
echo "Item 8: TG vs Residual Norm | seed=$SEED"
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

python -m experiments.run_ablation_residual_norm \
    --seed $SEED \
    --target_flops 0.5 \
    --ft_epochs 150 \
    --checkpoint_dir $SCRATCH/checkpoints \
    --data_path $SCRATCH/data \
    --save_dir $SCRATCH/checkpoints

echo "Completed: $(date)"
exit
