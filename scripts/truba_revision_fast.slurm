#!/bin/bash
# =============================================================================
# truba_revision_fast.slurm
# Items 10 + 11 — no training, ~2h
# SUBMIT: cd /arf/scratch/mhassan/transgap_v2 && sbatch /arf/home/mhassan/transgap_v2/scripts/truba_revision_fast.slurm
# =============================================================================
#SBATCH -J tg2_rev_fast
#SBATCH -A mhassan
#SBATCH -p akya-cuda
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 10
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/revision_fast_%j.out
#SBATCH --error=logs/revision_fast_%j.err

echo "============================================================"
echo "Job: $SLURM_JOB_ID on $SLURM_NODELIST"
echo "Items 10+11: Channel approx vs exact + Calibration subsets"
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

python -m experiments.run_ablation_channel_exact \
    --experiment both \
    --checkpoint_dir $SCRATCH/checkpoints \
    --data_path $SCRATCH/data \
    --save_dir $SCRATCH/checkpoints \
    --n_subsets 5

echo "Completed: $(date)"
exit
