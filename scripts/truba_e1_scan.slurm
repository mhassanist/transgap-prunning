#!/bin/bash
#SBATCH --job-name=tg_e1_scan
#SBATCH --account=mhassan
#SBATCH --partition=akya-cuda
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=0-00:30:00
#SBATCH --output=/arf/scratch/mhassan/transgap_v2/logs/e1_scan_%j.out
#SBATCH --error=/arf/scratch/mhassan/transgap_v2/logs/e1_scan_%j.err
#SBATCH --chdir=/arf/scratch/mhassan/transgap_v2

# E1: Scan — compute TG for all R110 blocks, save qualifying list
# Run this once before submitting per-block fine-tune jobs.
#
# Usage: sbatch scripts/truba_e1_scan.slurm

SCRATCH=/arf/scratch/mhassan/transgap_v2
CODE=/arf/home/mhassan/transgap_v2

mkdir -p $SCRATCH/{logs,checkpoints}

echo "============================================================"
echo "Job   : $SLURM_JOB_ID  on  $(hostname)"
echo "Task  : E1 TG scan (ResNet-110)"
echo "Start : $(date)"
echo "============================================================"

module purge
module load comp/python/miniconda3
source activate

python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
nvidia-smi --query-gpu=name,memory.total --format=csv 2>/dev/null

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

cd $CODE

python -m experiments.run_e1_r110_correlation \
    --scan \
    --checkpoint $SCRATCH/checkpoints/resnet110_cifar10_baseline.pth \
    --data_path  $SCRATCH/data \
    --save_dir   $SCRATCH/checkpoints

echo "============================================================"
echo "Scan done: $(date)"
echo "Next step: run launch_e1_r110.sh to submit per-block jobs"
echo "============================================================"
