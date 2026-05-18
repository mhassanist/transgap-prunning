#!/bin/bash
#SBATCH -J tg2_multi
#SBATCH -A mhassan
#SBATCH -p akya-cuda
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 10
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/multi_%j.out
#SBATCH --error=logs/multi_%j.err

# ============================================================
# TransGap v2 — Multi-Configuration Experiments
# ============================================================
# Experiments at different operating points to demonstrate:
#   - Block removal at various thresholds
#   - Channel pruning kicking in at aggressive targets
#   - Full multi-granularity pipeline
#
# SUBMIT FROM: cd /arf/scratch/$USER/transgap_v2 && sbatch /arf/home/$USER/transgap_v2/scripts/truba_multi.slurm

echo "============================================================"
echo "Job: $SLURM_JOB_ID on $SLURM_NODELIST"
echo "Start: $(date)"
echo "============================================================"

SCRATCH_DIR=/arf/scratch/$USER/transgap_v2
CODE_DIR=/arf/home/$USER/transgap_v2
CHECKPOINT=$SCRATCH_DIR/checkpoints/resnet56_cifar10_baseline.pth

mkdir -p $SCRATCH_DIR/logs
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
    exit 1
fi

cd $CODE_DIR

# ============================================================
# Exp 1: Moderate — 50% FLOPs, strict TG threshold (fewer blocks)
# Goal: Test block removal with tighter threshold + channel pruning
# ============================================================
echo ""
echo "============================================================"
echo "EXP 1: 50% FLOPs, max_block_tg=0.02 (strict, fewer blocks)"
echo "============================================================"
python -m experiments.run_pruning \
    --checkpoint $CHECKPOINT \
    --target_flops 0.5 \
    --max_block_tg 0.02 \
    --ft_epochs 300 \
    --ft_lr 0.01 \
    --data_path $SCRATCH_DIR/data \
    --save_dir $SCRATCH_DIR/checkpoints \
    --log_dir $SCRATCH_DIR/logs

# ============================================================
# Exp 2: Aggressive — 40% FLOPs remaining
# Goal: Force channel pruning to activate alongside block removal
# ============================================================
echo ""
echo "============================================================"
echo "EXP 2: 40% FLOPs remaining (aggressive)"
echo "============================================================"
python -m experiments.run_pruning \
    --checkpoint $CHECKPOINT \
    --target_flops 0.4 \
    --max_block_tg 0.05 \
    --ft_epochs 300 \
    --ft_lr 0.01 \
    --data_path $SCRATCH_DIR/data \
    --save_dir $SCRATCH_DIR/checkpoints \
    --log_dir $SCRATCH_DIR/logs

# ============================================================
# Exp 3: Very aggressive — 35% FLOPs remaining
# Goal: Push the limits, see where accuracy breaks down
# ============================================================
echo ""
echo "============================================================"
echo "EXP 3: 35% FLOPs remaining (very aggressive)"
echo "============================================================"
python -m experiments.run_pruning \
    --checkpoint $CHECKPOINT \
    --target_flops 0.35 \
    --max_block_tg 0.05 \
    --ft_epochs 300 \
    --ft_lr 0.01 \
    --data_path $SCRATCH_DIR/data \
    --save_dir $SCRATCH_DIR/checkpoints \
    --log_dir $SCRATCH_DIR/logs

# ============================================================
# Exp 4: Conservative — 60% FLOPs remaining, strict threshold
# Goal: High accuracy point with minimal pruning
# ============================================================
echo ""
echo "============================================================"
echo "EXP 4: 60% FLOPs remaining (conservative)"
echo "============================================================"
python -m experiments.run_pruning \
    --checkpoint $CHECKPOINT \
    --target_flops 0.6 \
    --max_block_tg 0.03 \
    --ft_epochs 300 \
    --ft_lr 0.01 \
    --data_path $SCRATCH_DIR/data \
    --save_dir $SCRATCH_DIR/checkpoints \
    --log_dir $SCRATCH_DIR/logs

echo ""
echo "============================================================"
echo "ALL EXPERIMENTS COMPLETED: $(date)"
echo "============================================================"

# Print all summaries
echo ""
echo "=== RESULTS SUMMARY ==="
for f in $SCRATCH_DIR/checkpoints/*_summary.json; do
    echo ""
    echo "--- $(basename $f) ---"
    cat $f | python -c "
import sys, json
d = json.load(sys.stdin)
print(f\"  Acc: {d['final_acc']:.2f}% (Δ{d['acc_delta']:+.2f}%)\")
print(f\"  FLOPs: {d['flops_reduction_pct']:.1f}% reduction\")
print(f\"  Blocks removed: {d['num_blocks_removed']}\")
print(f\"  Time: {d['training_time_hours']:.2f}h\")
" 2>/dev/null || echo "  (parse error)"
done

exit
