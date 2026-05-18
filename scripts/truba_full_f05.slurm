#!/bin/bash
#SBATCH -J tg2_fl05
#SBATCH -A mhassan
#SBATCH -p akya-cuda
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 10
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/full_f05_%j.out
#SBATCH --error=logs/full_f05_%j.err
# TransGap v2 — Full Pipeline, f0.5 only
# SUBMIT: cd /arf/scratch/$USER/transgap_v2 && sbatch /arf/home/$USER/transgap_v2/scripts/truba_full_f05.slurm
echo "===================================================="
echo "Job: $SLURM_JOB_ID on $SLURM_NODELIST | $(date)"
echo "===================================================="
SCRATCH_DIR=/arf/scratch/$USER/transgap_v2
CODE_DIR=/arf/home/$USER/transgap_v2
IMAGENET_DIR=/arf/scratch/$USER/imagenet
mkdir -p $SCRATCH_DIR/{logs,checkpoints}
cd $SCRATCH_DIR
module purge
module load comp/python/miniconda3
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
SUMMARY=$SCRATCH_DIR/checkpoints/resnet50_imagenet_full_f0.5_summary.json
if [ -f "$SUMMARY" ]; then
    echo "SKIP: full_f0.5 already completed"
    cat $SUMMARY | python -c "import sys,json; d=json.load(sys.stdin); print(f'  Top-1: {d[\"final_acc\"]:.2f}% ({d[\"acc_delta\"]:+.2f}%), FLOPs: {d[\"flops_reduction_pct\"]:.1f}%')" 2>/dev/null
    exit 0
fi
RESUME=$SCRATCH_DIR/checkpoints/resnet50_imagenet_full_f0.5_best_resume.pth
if [ -f "$RESUME" ]; then
    python -c "import torch; ck=torch.load('$RESUME',map_location='cpu',weights_only=False); print(f'Resuming ep {ck.get(\"epoch\",\"?\")}, best={ck.get(\"best_acc\",0):.2f}%')"
else
    echo "No resume checkpoint — starting from ep0"
fi
cd $CODE_DIR
python -m experiments.run_imagenet_full \
    --data_path $IMAGENET_DIR \
    --target_flops 0.5 \
    --max_blocks 3 \
    --ft_epochs 90 \
    --ft_lr 0.001 \
    --batch_size 64 \
    --save_dir $SCRATCH_DIR/checkpoints \
    --log_dir $SCRATCH_DIR/logs
if [ -f "$SUMMARY" ]; then
    echo "COMPLETED:"; cat $SUMMARY | python -c "import sys,json; d=json.load(sys.stdin); print(f'  Top-1: {d[\"final_acc\"]:.2f}% ({d[\"acc_delta\"]:+.2f}%), FLOPs: {d[\"flops_reduction_pct\"]:.1f}%')" 2>/dev/null
else
    echo "INCOMPLETE — resubmit: cd /arf/scratch/\$USER/transgap_v2 && sbatch /arf/home/\$USER/transgap_v2/scripts/truba_full_f05.slurm"
fi
