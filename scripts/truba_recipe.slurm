#!/bin/bash
#SBATCH -J tg2_recipe
#SBATCH -A mhassan
#SBATCH -p orfoz
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 10
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=/arf/scratch/mhassan/transgap_v2/logs/recipe_%j.out
#SBATCH --error=/arf/scratch/mhassan/transgap_v2/logs/recipe_%j.err

SETTING=${1:-r56_cifar10_f05}
CONDITION=${2:-full}

echo "Job: $SLURM_JOB_ID | Setting: $SETTING | Condition: $CONDITION | Start: $(date)"

SCRATCH=/arf/scratch/mhassan/transgap_v2
CODE=/arf/home/mhassan/transgap_v2
mkdir -p $SCRATCH/logs $SCRATCH/checkpoints
cd $CODE

module purge
module load comp/python/miniconda3
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

python -m experiments.run_ablation_recipe \
    --setting $SETTING \
    --condition $CONDITION \
    --checkpoint_dir $SCRATCH/checkpoints \
    --data_path $SCRATCH/data \
    --save_dir $SCRATCH/checkpoints

echo "Completed: $(date)"
