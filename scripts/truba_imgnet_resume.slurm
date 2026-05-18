#!/bin/bash
#SBATCH -J tg2_resume
#SBATCH -A mhassan
#SBATCH -p akya-cuda
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 10
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/imgnet_resume_%j.out
#SBATCH --error=logs/imgnet_resume_%j.err

# ============================================================
# TransGap v2 — Resume ImageNet f0.6 experiment
# ============================================================
# Resumes from epoch 49 checkpoint, finishes epochs 50-90.
# ~23 epochs × ~63 min/epoch ≈ 24 hours
#
# SUBMIT FROM: cd /arf/scratch/$USER/transgap_v2 && sbatch /arf/home/$USER/transgap_v2/scripts/truba_imgnet_resume.slurm

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
echo ""
echo "Resuming f0.6 experiment..."
echo "Resume checkpoint: $SCRATCH_DIR/checkpoints/resnet50_imagenet_f0.6_best_resume.pth"
ls -la $SCRATCH_DIR/checkpoints/resnet50_imagenet_f0.6_*

cd $CODE_DIR

python -m experiments.run_imagenet \
    --data_path $IMAGENET_DIR \
    --target_flops 0.6 \
    --max_blocks 2 \
    --ft_epochs 90 \
    --ft_lr 0.001 \
    --batch_size 64 \
    --save_dir $SCRATCH_DIR/checkpoints \
    --log_dir $SCRATCH_DIR/logs

echo ""
echo "============================================================"
echo "RESUME COMPLETE: $(date)"
echo "============================================================"

# Print result
SUMMARY=$SCRATCH_DIR/checkpoints/resnet50_imagenet_f0.6_summary.json
if [ -f "$SUMMARY" ]; then
    echo "Result:"
    cat $SUMMARY | python -c "
import sys, json
d = json.load(sys.stdin)
print(f\"  Top-1: {d['final_acc']:.2f}% (delta {d['acc_delta']:+.2f}%)\")
print(f\"  FLOPs: {d['flops_reduction_pct']:.1f}% reduction\")
print(f\"  Time: {d['training_time_hours']:.2f}h\")
" 2>/dev/null
fi

exit
