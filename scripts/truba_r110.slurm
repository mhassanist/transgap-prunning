#!/bin/bash
#SBATCH -J tg2_r110
#SBATCH -A mhassan
#SBATCH -p akya-cuda
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 10
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/r110_%j.out
#SBATCH --error=logs/r110_%j.err

# ============================================================
# TransGap v2 — ResNet-110 / CIFAR-10 Full Pipeline
# ============================================================
# Step 1: Train baseline (~2h for 300 epochs)
# Step 2: Run 4 pruning experiments (~1h each)
# Total: ~6 hours
#
# SUBMIT FROM: cd /arf/scratch/$USER/transgap_v2 && sbatch /arf/home/$USER/transgap_v2/scripts/truba_r110.slurm

echo "============================================================"
echo "Job: $SLURM_JOB_ID on $SLURM_NODELIST"
echo "Start: $(date)"
echo "============================================================"

SCRATCH_DIR=/arf/scratch/$USER/transgap_v2
CODE_DIR=/arf/home/$USER/transgap_v2
CHECKPOINT=$SCRATCH_DIR/checkpoints/resnet110_cifar10_baseline.pth

mkdir -p $SCRATCH_DIR/{logs,checkpoints}
cd $SCRATCH_DIR

module purge
module load comp/python/miniconda3

echo "Python: $(which python)"
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
nvidia-smi --query-gpu=name,memory.total --format=csv 2>/dev/null

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

cd $CODE_DIR

# ============================================================
# STEP 1: Baseline Training (resumable)
# ============================================================
echo ""
echo "============================================================"
echo "STEP 1: ResNet-110 Baseline Training"
echo "============================================================"
python -m experiments.run_baseline \
    --model resnet110 \
    --config configs/baseline_cifar10_r110.yaml \
    --data_path $SCRATCH_DIR/data \
    --save_dir $SCRATCH_DIR/checkpoints \
    --log_dir $SCRATCH_DIR/logs

# Verify baseline exists
if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: Baseline training failed — checkpoint not found"
    exit 1
fi
echo "Baseline checkpoint: $CHECKPOINT"

# ============================================================
# STEP 2: Pruning Experiments
# ============================================================

# Exp R110-1: Conservative (60% FLOPs, strict TG)
echo ""
echo "============================================================"
echo "EXP R110-1: 60% FLOPs, max_block_tg=0.03 (conservative)"
echo "============================================================"
python -m experiments.run_pruning \
    --checkpoint $CHECKPOINT \
    --model resnet110 \
    --dataset cifar10 \
    --target_flops 0.6 \
    --max_block_tg 0.03 \
    --ft_epochs 300 \
    --ft_lr 0.01 \
    --data_path $SCRATCH_DIR/data \
    --save_dir $SCRATCH_DIR/checkpoints \
    --log_dir $SCRATCH_DIR/logs

# Exp R110-2: Multi-granularity demo (50% FLOPs, strict TG)
echo ""
echo "============================================================"
echo "EXP R110-2: 50% FLOPs, max_block_tg=0.02 (multi-granularity)"
echo "============================================================"
python -m experiments.run_pruning \
    --checkpoint $CHECKPOINT \
    --model resnet110 \
    --dataset cifar10 \
    --target_flops 0.5 \
    --max_block_tg 0.02 \
    --ft_epochs 300 \
    --ft_lr 0.01 \
    --data_path $SCRATCH_DIR/data \
    --save_dir $SCRATCH_DIR/checkpoints \
    --log_dir $SCRATCH_DIR/logs

# Exp R110-3: Aggressive (40% FLOPs)
echo ""
echo "============================================================"
echo "EXP R110-3: 40% FLOPs remaining (aggressive)"
echo "============================================================"
python -m experiments.run_pruning \
    --checkpoint $CHECKPOINT \
    --model resnet110 \
    --dataset cifar10 \
    --target_flops 0.4 \
    --max_block_tg 0.05 \
    --ft_epochs 300 \
    --ft_lr 0.01 \
    --data_path $SCRATCH_DIR/data \
    --save_dir $SCRATCH_DIR/checkpoints \
    --log_dir $SCRATCH_DIR/logs

# Exp R110-4: Very aggressive (35% FLOPs)
echo ""
echo "============================================================"
echo "EXP R110-4: 35% FLOPs remaining (very aggressive)"
echo "============================================================"
python -m experiments.run_pruning \
    --checkpoint $CHECKPOINT \
    --model resnet110 \
    --dataset cifar10 \
    --target_flops 0.35 \
    --max_block_tg 0.05 \
    --ft_epochs 300 \
    --ft_lr 0.01 \
    --data_path $SCRATCH_DIR/data \
    --save_dir $SCRATCH_DIR/checkpoints \
    --log_dir $SCRATCH_DIR/logs

echo ""
echo "============================================================"
echo "ALL ResNet-110 EXPERIMENTS COMPLETED: $(date)"
echo "============================================================"

# Print summaries
echo ""
echo "=== RESNET-110 RESULTS SUMMARY ==="
for f in $SCRATCH_DIR/checkpoints/resnet110*_summary.json; do
    echo ""
    echo "--- $(basename $f) ---"
    cat $f | python -c "
import sys, json
d = json.load(sys.stdin)
print(f\"  Acc: {d['final_acc']:.2f}% (delta {d['acc_delta']:+.2f}%)\")
print(f\"  FLOPs: {d['flops_reduction_pct']:.1f}% reduction\")
print(f\"  Blocks removed: {d['num_blocks_removed']}\")
print(f\"  Channel ratios: {len(d['channel_ratios'])} blocks pruned\")
print(f\"  Time: {d['training_time_hours']:.2f}h\")
" 2>/dev/null || echo "  (parse error)"
done

exit
