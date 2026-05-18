#!/bin/bash
# =============================================================================
# truba_revision_recipe.slurm
# Item 9 — KD+Mixup recipe ablation, ~12h (4 conditions × 150 epochs)
# SUBMIT: cd /arf/scratch/mhassan/transgap_v2 && sbatch /arf/home/mhassan/transgap_v2/scripts/truba_revision_recipe.slurm [condition]
# condition: full | no_kd | no_mixup | neither | all  (default: all)
# =============================================================================
#SBATCH -J tg2_rev_recipe
#SBATCH -A mhassan
#SBATCH -p akya-cuda
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 10
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/revision_recipe_%j.out
#SBATCH --error=logs/revision_recipe_%j.err

CONDITION=${1:-all}

echo "============================================================"
echo "Job: $SLURM_JOB_ID on $SLURM_NODELIST"
echo "Item 9: Recipe ablation | condition=$CONDITION"
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

python -m experiments.run_ablation_recipe \
    --condition $CONDITION \
    --ft_epochs 150 \
    --checkpoint_dir $SCRATCH/checkpoints \
    --data_path $SCRATCH/data \
    --save_dir $SCRATCH/checkpoints

echo "Completed: $(date)"
exit
