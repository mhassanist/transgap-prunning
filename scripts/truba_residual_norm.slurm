#!/bin/bash
#SBATCH -J tg2_resnorm
#SBATCH -A mhassan
#SBATCH -p orfoz
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 10
#SBATCH --gres=gpu:1
#SBATCH --time=16:00:00
#SBATCH --output=/arf/scratch/mhassan/transgap_v2/logs/residual_norm_%j.out
#SBATCH --error=/arf/scratch/mhassan/transgap_v2/logs/residual_norm_%j.err

SEED=${1:-42}

echo "Job: $SLURM_JOB_ID | Seed: $SEED | Start: $(date)"

SCRATCH=/arf/scratch/mhassan/transgap_v2
CODE=/arf/home/mhassan/transgap_v2
mkdir -p $SCRATCH/logs $SCRATCH/checkpoints
cd $CODE

module purge
module load comp/python/miniconda3
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

python -m experiments.run_ablation_residual_norm \
    --seed $SEED \
    --target_flops 0.5 \
    --ft_epochs 150 \
    --checkpoint_dir $SCRATCH/checkpoints \
    --data_path $SCRATCH/data \
    --save_dir $SCRATCH/checkpoints

echo "Completed: $(date)"
