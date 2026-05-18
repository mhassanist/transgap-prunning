#!/bin/bash
#SBATCH -J tg2_orthog
#SBATCH -A mhassan
#SBATCH -p orfoz
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 10
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=/arf/scratch/mhassan/transgap_v2/logs/orthogonality_%j.out
#SBATCH --error=/arf/scratch/mhassan/transgap_v2/logs/orthogonality_%j.err

BACKBONE=${1:-all}

echo "Job: $SLURM_JOB_ID | Backbone: $BACKBONE | Start: $(date)"

SCRATCH=/arf/scratch/mhassan/transgap_v2
CODE=/arf/home/mhassan/transgap_v2
mkdir -p $SCRATCH/logs $SCRATCH/checkpoints
cd $CODE

module purge
module load comp/python/miniconda3
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

python -m experiments.run_orthogonality \
    --backbone $BACKBONE \
    --checkpoint_dir $SCRATCH/checkpoints \
    --data_path $SCRATCH/data \
    --imagenet_path /arf/scratch/mhassan/imagenet \
    --save_dir $SCRATCH/checkpoints

echo "Completed: $(date)"
