#!/bin/bash
#SBATCH -J tg2_imgn2
#SBATCH -A mhassan
#SBATCH -p akya-cuda
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 10
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/imagenet2_%j.out
#SBATCH --error=logs/imagenet2_%j.err

# ============================================================
# TransGap v2 — ImageNet / ResNet-50 (v2 Fixed)
# ============================================================
# Changes from v1:
#   1. No hard TG threshold — greedy block selection
#   2. Channel pruning as primary compression for Bottleneck
#   3. TG-guided channel ratios (lower TG → prune more)
#   4. max_blocks limits block removal (default 4)
#
# 3 experiments: 60%, 50%, 40% FLOPs remaining
# Each ~8h on V100 → total ~24h
#
# SUBMIT: cd /arf/scratch/$USER/transgap_v2 && sbatch /arf/home/$USER/transgap_v2/scripts/truba_imagenet2.slurm

echo "============================================================"
echo "Job: $SLURM_JOB_ID on $SLURM_NODELIST"
echo "Start: $(date)"
echo "============================================================"

SCRATCH_DIR=/arf/scratch/$USER/transgap_v2
CODE_DIR=/arf/home/$USER/transgap_v2
IMAGENET_DIR=/arf/scratch/$USER/imagenet

mkdir -p $SCRATCH_DIR/{logs,checkpoints}
cd $SCRATCH_DIR

module purge
module load comp/python/miniconda3

echo "Python: $(which python)"
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
nvidia-smi --query-gpu=name,memory.total --format=csv 2>/dev/null

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

# Verify ImageNet
if [ ! -d "$IMAGENET_DIR/train" ] || [ ! -d "$IMAGENET_DIR/val" ]; then
    echo "ERROR: ImageNet data not found at $IMAGENET_DIR"
    exit 1
fi
echo "ImageNet data: $IMAGENET_DIR"

# Clean up old failed experiment files
echo ""
echo "Cleaning up old failed experiment files..."
rm -f $SCRATCH_DIR/checkpoints/resnet50_imagenet_pruned_b0_f0.6* 2>/dev/null
rm -f $SCRATCH_DIR/checkpoints/resnet50_imagenet_pruned_b0_f0.5* 2>/dev/null
rm -f $SCRATCH_DIR/checkpoints/resnet50_imagenet_pruned_b0_f0.4* 2>/dev/null
rm -f $SCRATCH_DIR/checkpoints/resnet50_imagenet_f*_pruned_state.pth 2>/dev/null
rm -f $SCRATCH_DIR/checkpoints/resnet50_imagenet_f*_resume.pth 2>/dev/null
echo "Done."

cd $CODE_DIR

# ============================================================
# Run 3 experiments — skip if summary already exists
# ============================================================

# Experiment configs: target_flops max_blocks
# Conservative: 60% remaining, up to 2 blocks
# Standard:     50% remaining, up to 4 blocks  
# Aggressive:   40% remaining, up to 4 blocks

declare -a FLOPS_TARGETS=(0.6 0.5 0.4)
declare -a MAX_BLOCKS=(2 4 4)

for i in 0 1 2; do
    FLOPS=${FLOPS_TARGETS[$i]}
    BLOCKS=${MAX_BLOCKS[$i]}
    
    SUMMARY=$SCRATCH_DIR/checkpoints/resnet50_imagenet_f${FLOPS}_summary.json
    
    if [ -f "$SUMMARY" ]; then
        echo ""
        echo "============================================================"
        echo "SKIP: target_flops=$FLOPS — already completed"
        echo "============================================================"
        cat $SUMMARY | python -c "
import sys, json
d = json.load(sys.stdin)
print(f\"  Top-1: {d['final_acc']:.2f}% (delta {d['acc_delta']:+.2f}%)\")
print(f\"  FLOPs: {d['flops_reduction_pct']:.1f}% reduction\")
print(f\"  Blocks: {d['num_blocks_removed']}, Time: {d['training_time_hours']:.2f}h\")
" 2>/dev/null
        continue
    fi
    
    echo ""
    echo "============================================================"
    echo "ImageNet Exp $((i+1))/3: target_flops=$FLOPS, max_blocks=$BLOCKS"
    echo "============================================================"
    python -m experiments.run_imagenet \
        --data_path $IMAGENET_DIR \
        --target_flops $FLOPS \
        --max_blocks $BLOCKS \
        --ft_epochs 90 \
        --ft_lr 0.001 \
        --batch_size 64 \
        --save_dir $SCRATCH_DIR/checkpoints \
        --log_dir $SCRATCH_DIR/logs
done

echo ""
echo "============================================================"
echo "IMAGENET EXPERIMENTS STATUS: $(date)"
echo "============================================================"

DONE=0
TOTAL=3
for FLOPS in 0.6 0.5 0.4; do
    SUMMARY=$SCRATCH_DIR/checkpoints/resnet50_imagenet_f${FLOPS}_summary.json
    if [ -f "$SUMMARY" ]; then
        DONE=$((DONE + 1))
        echo "[DONE] target_flops=$FLOPS"
        cat $SUMMARY | python -c "
import sys, json
d = json.load(sys.stdin)
print(f\"  Top-1: {d['final_acc']:.2f}% ({d['acc_delta']:+.2f}%), FLOPs: {d['flops_reduction_pct']:.1f}%, Blocks: {d['num_blocks_removed']}, Time: {d['training_time_hours']:.2f}h\")
" 2>/dev/null
    else
        echo "[PENDING] target_flops=$FLOPS"
    fi
done

echo ""
echo "Completed: $DONE / $TOTAL"
if [ $DONE -lt $TOTAL ]; then
    echo "RESUBMIT: cd /arf/scratch/\$USER/transgap_v2 && sbatch /arf/home/\$USER/transgap_v2/scripts/truba_imagenet2.slurm"
fi

exit
