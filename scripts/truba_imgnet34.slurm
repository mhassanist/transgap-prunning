#!/bin/bash
#SBATCH -J tg2_img34
#SBATCH -A mhassan
#SBATCH -p akya-cuda
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 10
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/imgnet34_%j.out
#SBATCH --error=logs/imgnet34_%j.err

# ============================================================
# TransGap v2 — ImageNet Exp 2 (f0.5) and Exp 3 (f0.4)
# ============================================================
# f0.6 already completed: 76.17% (+0.04%) at 41.6% FLOPs reduction
# Now running the higher compression targets.
#
# f0.5: 50% FLOPs remaining → ~50% reduction, max 3 blocks
# f0.4: 40% FLOPs remaining → ~60% reduction, max 4 blocks
#
# Each ~90 epochs × ~63 min/epoch ≈ 95h per experiment
# Total: ~190h (within 3-day limit for 1, resubmit for 2nd)
#
# SUBMIT FROM: cd /arf/scratch/$USER/transgap_v2 && sbatch /arf/home/$USER/transgap_v2/scripts/truba_imgnet34.slurm

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

echo "ImageNet data: $IMAGENET_DIR"

cd $CODE_DIR

# ============================================================
# Exp 2: 50% FLOPs remaining (max 3 blocks)
# ============================================================
SUMMARY=$SCRATCH_DIR/checkpoints/resnet50_imagenet_f0.5_summary.json
if [ -f "$SUMMARY" ]; then
    echo ""
    echo "SKIP: f0.5 already completed"
    cat $SUMMARY | python -c "
import sys, json
d = json.load(sys.stdin)
print(f'  Top-1: {d[\"final_acc\"]:.2f}% (delta {d[\"acc_delta\"]:+.2f}%), FLOPs: {d[\"flops_reduction_pct\"]:.1f}%')
" 2>/dev/null
else
    echo ""
    echo "============================================================"
    echo "ImageNet Exp 2: target_flops=0.5, max_blocks=3"
    echo "============================================================"
    python -m experiments.run_imagenet \
        --data_path $IMAGENET_DIR \
        --target_flops 0.5 \
        --max_blocks 3 \
        --ft_epochs 90 \
        --ft_lr 0.001 \
        --batch_size 64 \
        --save_dir $SCRATCH_DIR/checkpoints \
        --log_dir $SCRATCH_DIR/logs
fi

# ============================================================
# Exp 3: 40% FLOPs remaining (max 4 blocks)
# ============================================================
SUMMARY=$SCRATCH_DIR/checkpoints/resnet50_imagenet_f0.4_summary.json
if [ -f "$SUMMARY" ]; then
    echo ""
    echo "SKIP: f0.4 already completed"
    cat $SUMMARY | python -c "
import sys, json
d = json.load(sys.stdin)
print(f'  Top-1: {d[\"final_acc\"]:.2f}% (delta {d[\"acc_delta\"]:+.2f}%), FLOPs: {d[\"flops_reduction_pct\"]:.1f}%')
" 2>/dev/null
else
    echo ""
    echo "============================================================"
    echo "ImageNet Exp 3: target_flops=0.4, max_blocks=4"
    echo "============================================================"
    python -m experiments.run_imagenet \
        --data_path $IMAGENET_DIR \
        --target_flops 0.4 \
        --max_blocks 4 \
        --ft_epochs 90 \
        --ft_lr 0.001 \
        --batch_size 64 \
        --save_dir $SCRATCH_DIR/checkpoints \
        --log_dir $SCRATCH_DIR/logs
fi

# ============================================================
# Status
# ============================================================
echo ""
echo "============================================================"
echo "IMAGENET STATUS: $(date)"
echo "============================================================"

for FLOPS in 0.6 0.5 0.4; do
    SUMMARY=$SCRATCH_DIR/checkpoints/resnet50_imagenet_f${FLOPS}_summary.json
    if [ -f "$SUMMARY" ]; then
        echo -n "[DONE] f=$FLOPS: "
        cat $SUMMARY | python -c "
import sys, json
d = json.load(sys.stdin)
print(f'Top-1: {d[\"final_acc\"]:.2f}% ({d[\"acc_delta\"]:+.2f}%), FLOPs red: {d[\"flops_reduction_pct\"]:.1f}%')
" 2>/dev/null
    else
        echo "[PENDING] f=$FLOPS — resubmit to continue"
    fi
done

exit
