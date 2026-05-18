#!/bin/bash
#SBATCH -J tg2_imgnet
#SBATCH -A mhassan
#SBATCH -p akya-cuda
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 10
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/imagenet_%j.out
#SBATCH --error=logs/imagenet_%j.err

# ============================================================
# TransGap v2 — ImageNet / ResNet-50 (Sequential, Resumable)
# ============================================================
# Runs 3 experiments sequentially. Each experiment saves periodic
# checkpoints every 10 epochs. If the job times out, resubmit
# the SAME script — it will skip completed experiments and
# resume any interrupted ones automatically.
#
# SUBMIT FROM: cd /arf/scratch/$USER/transgap_v2 && sbatch /arf/home/$USER/transgap_v2/scripts/truba_imagenet.slurm

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

# Verify ImageNet data
if [ ! -d "$IMAGENET_DIR/train" ] || [ ! -d "$IMAGENET_DIR/val" ]; then
    echo "ERROR: ImageNet data not found at $IMAGENET_DIR"
    exit 1
fi
echo "ImageNet data: $IMAGENET_DIR"

cd $CODE_DIR

# ============================================================
# Run experiments — skip if summary already exists (= completed)
# ============================================================

for FLOPS in 0.6 0.5 0.4; do
    SUMMARY=$SCRATCH_DIR/checkpoints/resnet50_imagenet_pruned_*_f${FLOPS}_summary.json

    # Check if this experiment is already done
    if ls $SUMMARY 1>/dev/null 2>&1; then
        echo ""
        echo "============================================================"
        echo "SKIP: ImageNet target_flops=$FLOPS — already completed"
        echo "============================================================"
        cat $SUMMARY 2>/dev/null | python -c "
import sys, json
d = json.load(sys.stdin)
print(f\"  Result: {d['final_acc']:.2f}% ({d['acc_delta']:+.2f}%), {d['flops_reduction_pct']:.1f}% FLOPs red.\")
" 2>/dev/null
        continue
    fi

    echo ""
    echo "============================================================"
    echo "ImageNet: target_flops=$FLOPS"
    echo "============================================================"
    python -m experiments.run_imagenet \
        --data_path $IMAGENET_DIR \
        --target_flops $FLOPS \
        --max_block_tg 0.05 \
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
    SUMMARY=$SCRATCH_DIR/checkpoints/resnet50_imagenet_pruned_*_f${FLOPS}_summary.json
    if ls $SUMMARY 1>/dev/null 2>&1; then
        DONE=$((DONE + 1))
        echo "[DONE] target_flops=$FLOPS"
        cat $SUMMARY 2>/dev/null | python -c "
import sys, json
d = json.load(sys.stdin)
print(f\"  Top-1: {d['final_acc']:.2f}% (delta {d['acc_delta']:+.2f}%), FLOPs red: {d['flops_reduction_pct']:.1f}%, Blocks: {d['num_blocks_removed']}, Time: {d['training_time_hours']:.2f}h\")
" 2>/dev/null
    else
        echo "[PENDING] target_flops=$FLOPS — resubmit this script to continue"
    fi
done

echo ""
echo "Completed: $DONE / $TOTAL"
if [ $DONE -lt $TOTAL ]; then
    echo "RESUBMIT: cd /arf/scratch/\$USER/transgap_v2 && sbatch /arf/home/\$USER/transgap_v2/scripts/truba_imagenet.slurm"
fi

exit
