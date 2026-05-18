#!/bin/bash
#SBATCH -J tg2_chanexact
#SBATCH -A mhassan
#SBATCH -p orfoz
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 10
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=/arf/scratch/mhassan/transgap_v2/logs/channel_exact_%j.out
#SBATCH --error=/arf/scratch/mhassan/transgap_v2/logs/channel_exact_%j.err

echo "Job: $SLURM_JOB_ID | Items 10+11 | Start: $(date)"

SCRATCH=/arf/scratch/mhassan/transgap_v2
CODE=/arf/home/mhassan/transgap_v2
mkdir -p $SCRATCH/logs $SCRATCH/checkpoints
cd $CODE

module purge
module load comp/python/miniconda3
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

python -m experiments.run_ablation_channel_exact \
    --experiment both \
    --checkpoint_dir $SCRATCH/checkpoints \
    --data_path $SCRATCH/data \
    --save_dir $SCRATCH/checkpoints \
    --n_subsets 5

echo "Completed: $(date)"
