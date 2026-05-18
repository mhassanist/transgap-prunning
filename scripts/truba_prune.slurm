#!/bin/bash
#SBATCH -J tg2_prune
#SBATCH -A mhassan
#SBATCH -p akya-cuda
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 10
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/prune_%j.out
#SBATCH --error=logs/prune_%j.err

# ============================================
# TransGap v2 — Pruning Pipeline (Resumable)
# ============================================
# SUBMIT FROM: cd /arf/scratch/$USER/transgap_v2 && sbatch /arf/home/$USER/transgap_v2/scripts/truba_prune.slurm

echo "============================================"
echo "Job: $SLURM_JOB_ID on $SLURM_NODELIST"
echo "Start: $(date)"
echo "============================================"

SCRATCH_DIR=/arf/scratch/$USER/transgap_v2
CODE_DIR=/arf/home/$USER/transgap_v2
CHECKPOINT=$SCRATCH_DIR/checkpoints/resnet56_cifar10_baseline.pth

mkdir -p $SCRATCH_DIR/{logs,checkpoints}
cd $SCRATCH_DIR

module purge
module load comp/python/miniconda3

echo "Python: $(which python)"
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
nvidia-smi --query-gpu=name,memory.total --format=csv 2>/dev/null

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: Baseline checkpoint not found: $CHECKPOINT"
    echo "Run truba_baseline.slurm first!"
    exit 1
fi

cd $CODE_DIR

python -m experiments.run_pruning \
    --checkpoint $CHECKPOINT \
    --config configs/prune_cifar10.yaml \
    --target_flops 0.5 \
    --data_path $SCRATCH_DIR/data \
    --save_dir $SCRATCH_DIR/checkpoints \
    --log_dir $SCRATCH_DIR/logs

echo "============================================"
echo "Completed: $(date)"
echo "============================================"
exit
