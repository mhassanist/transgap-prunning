#!/bin/bash
#SBATCH --job-name=tg2_a8
#SBATCH --partition=akya-cuda
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=/arf/scratch/mhassan/transgap_v2/logs/a8full_%j.out
#SBATCH --error=/arf/scratch/mhassan/transgap_v2/logs/a8full_%j.err
#SBATCH --chdir=/arf/scratch/mhassan/transgap_v2

# Usage: sbatch truba_ablation_A8.slurm <backbone> <block>
# e.g.   sbatch truba_ablation_A8.slurm r56_cifar10 layer1.5
#        sbatch truba_ablation_A8.slurm r50_imagenet layer1.1

BACKBONE=$1
BLOCK=$2

module load comp/python/miniconda3
source activate

echo "=============================================="
echo "Job: $SLURM_JOB_ID on $(hostname)"
echo "A8 single-block removal | backbone=$BACKBONE block=$BLOCK"
echo "Start: $(date)"
echo "=============================================="

nvidia-smi --query-gpu=name,memory.total --format=csv

cd /arf/home/mhassan/transgap_v2
python -m experiments.run_ablation_A8_full \
    --backbone $BACKBONE \
    --block $BLOCK \
    --save_dir /arf/scratch/mhassan/transgap_v2/checkpoints
STATUS=$?

if [ $STATUS -eq 0 ]; then
    echo "SUCCEEDED: $(date)"
else
    echo "FAILED (exit code $STATUS): $(date) -- see the .err log for the traceback"
fi
exit $STATUS
