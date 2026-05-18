#!/bin/bash
#SBATCH -J tg2_full
#SBATCH -A mhassan
#SBATCH -p akya-cuda
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 10
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/imgnet_full_%j.out
#SBATCH --error=logs/imgnet_full_%j.err

# ============================================================
# TransGap v2 — ImageNet Full Pipeline
# ============================================================
# Runs the COMPLETE pipeline matching the paper diagram:
#   Stage 1: TG Computation
#   Stage 2: Safety Bounds (Lipschitz + DPI)
#   Stage 3: Joint Optimization
#   Stage 4: Block + Channel Pruning
#   Stage 5: KD + TG-Scaled LR + Mixup
#
# Output files: resnet50_imagenet_full_f{X}_*
# Does NOT conflict with running experiments (different filenames).
#
# 3 experiments: f0.6 (2 blocks), f0.5 (3 blocks), f0.4 (4 blocks)
# Each ~95h, so will need resubmission.
#
# SUBMIT FROM: cd /arf/scratch/$USER/transgap_v2 && sbatch /arf/home/$USER/transgap_v2/scripts/truba_imgnet_full.slurm

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
echo "Pipeline: FULL (Safety Bounds + TG-Scaled LR + Mixup)"

cd $CODE_DIR

# Experiment configs: target_flops max_blocks
declare -a FLOPS_TARGETS=(0.6 0.5 0.4)
declare -a MAX_BLOCKS=(2 3 4)

for i in 0 1 2; do
    FLOPS=${FLOPS_TARGETS[$i]}
    BLOCKS=${MAX_BLOCKS[$i]}
    
    SUMMARY=$SCRATCH_DIR/checkpoints/resnet50_imagenet_full_f${FLOPS}_summary.json
    
    if [ -f "$SUMMARY" ]; then
        echo ""
        echo "SKIP: full_f$FLOPS already completed"
        cat $SUMMARY | python -c "
import sys, json
d = json.load(sys.stdin)
print(f'  Top-1: {d[\"final_acc\"]:.2f}% ({d[\"acc_delta\"]:+.2f}%), FLOPs: {d[\"flops_reduction_pct\"]:.1f}%')
" 2>/dev/null
        continue
    fi
    
    echo ""
    echo "============================================================"
    echo "Full Pipeline Exp: target_flops=$FLOPS, max_blocks=$BLOCKS"
    echo "============================================================"
    python -m experiments.run_imagenet_full \
        --data_path $IMAGENET_DIR \
        --target_flops $FLOPS \
        --max_blocks $BLOCKS \
        --ft_epochs 90 \
        --ft_lr 0.001 \
        --batch_size 64 \
        --save_dir $SCRATCH_DIR/checkpoints \
        --log_dir $SCRATCH_DIR/logs
done

# Status
echo ""
echo "============================================================"
echo "FULL PIPELINE STATUS: $(date)"
echo "============================================================"

for FLOPS in 0.6 0.5 0.4; do
    SUMMARY=$SCRATCH_DIR/checkpoints/resnet50_imagenet_full_f${FLOPS}_summary.json
    if [ -f "$SUMMARY" ]; then
        echo -n "[DONE] full_f=$FLOPS: "
        cat $SUMMARY | python -c "
import sys, json
d = json.load(sys.stdin)
print(f'Top-1: {d[\"final_acc\"]:.2f}% ({d[\"acc_delta\"]:+.2f}%), FLOPs: {d[\"flops_reduction_pct\"]:.1f}%')
" 2>/dev/null
    else
        echo "[PENDING] full_f=$FLOPS — resubmit to continue"
    fi
done

echo ""
echo "Resubmit: cd /arf/scratch/\$USER/transgap_v2 && sbatch /arf/home/\$USER/transgap_v2/scripts/truba_imgnet_full.slurm"

exit
