#!/bin/bash
#SBATCH -J tg2_c100_abl
#SBATCH -A mhassan
#SBATCH -p akya-cuda
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 10
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/c100_abl_%j.out
#SBATCH --error=logs/c100_abl_%j.err

# ============================================================
# TransGap v2 — CIFAR-100 + Ablations
# ============================================================
# Part A: CIFAR-100 baselines (R56 + R110) + pruning experiments
# Part B: Ablations (A1, A5, A7, A8) on CIFAR-10/ResNet-56
#
# SUBMIT FROM: cd /arf/scratch/$USER/transgap_v2 && sbatch /arf/home/$USER/transgap_v2/scripts/truba_c100_ablations.slurm

echo "============================================================"
echo "Job: $SLURM_JOB_ID on $SLURM_NODELIST"
echo "Start: $(date)"
echo "============================================================"

SCRATCH_DIR=/arf/scratch/$USER/transgap_v2
CODE_DIR=/arf/home/$USER/transgap_v2

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
# PART A: CIFAR-100
# ============================================================

# === A.1: CIFAR-100 / ResNet-56 Baseline ===
echo ""
echo "============================================================"
echo "CIFAR-100 / ResNet-56 BASELINE"
echo "============================================================"
python -m experiments.run_baseline \
    --model resnet56 \
    --dataset cifar100 \
    --epochs 300 \
    --data_path $SCRATCH_DIR/data \
    --save_dir $SCRATCH_DIR/checkpoints \
    --log_dir $SCRATCH_DIR/logs

C100_R56=$SCRATCH_DIR/checkpoints/resnet56_cifar100_baseline.pth

# === A.2: CIFAR-100 / ResNet-56 Pruning ===
if [ -f "$C100_R56" ]; then
    for FLOPS in 0.6 0.5 0.4; do
        echo ""
        echo "============================================================"
        echo "CIFAR-100 / ResNet-56 / Target FLOPs = $FLOPS"
        echo "============================================================"
        python -m experiments.run_pruning \
            --checkpoint $C100_R56 \
            --model resnet56 \
            --dataset cifar100 \
            --target_flops $FLOPS \
            --max_block_tg 0.05 \
            --ft_epochs 300 \
            --ft_lr 0.01 \
            --data_path $SCRATCH_DIR/data \
            --save_dir $SCRATCH_DIR/checkpoints \
            --log_dir $SCRATCH_DIR/logs
    done
fi

# === A.3: CIFAR-100 / ResNet-110 Baseline ===
echo ""
echo "============================================================"
echo "CIFAR-100 / ResNet-110 BASELINE"
echo "============================================================"
python -m experiments.run_baseline \
    --model resnet110 \
    --dataset cifar100 \
    --epochs 300 \
    --data_path $SCRATCH_DIR/data \
    --save_dir $SCRATCH_DIR/checkpoints \
    --log_dir $SCRATCH_DIR/logs

C100_R110=$SCRATCH_DIR/checkpoints/resnet110_cifar100_baseline.pth

# === A.4: CIFAR-100 / ResNet-110 Pruning ===
if [ -f "$C100_R110" ]; then
    for FLOPS in 0.6 0.5 0.4; do
        echo ""
        echo "============================================================"
        echo "CIFAR-100 / ResNet-110 / Target FLOPs = $FLOPS"
        echo "============================================================"
        python -m experiments.run_pruning \
            --checkpoint $C100_R110 \
            --model resnet110 \
            --dataset cifar100 \
            --target_flops $FLOPS \
            --max_block_tg 0.05 \
            --ft_epochs 300 \
            --ft_lr 0.01 \
            --data_path $SCRATCH_DIR/data \
            --save_dir $SCRATCH_DIR/checkpoints \
            --log_dir $SCRATCH_DIR/logs
    done
fi

# ============================================================
# PART B: ABLATIONS (on CIFAR-10 / ResNet-56)
# ============================================================

C10_R56=$SCRATCH_DIR/checkpoints/resnet56_cifar10_baseline.pth

if [ ! -f "$C10_R56" ]; then
    echo "WARNING: CIFAR-10/ResNet-56 baseline not found at $C10_R56"
    echo "Checking code dir..."
    if [ -f "/arf/home/$USER/transgap_v2/checkpoints/resnet56_cifar10_baseline.pth" ]; then
        cp /arf/home/$USER/transgap_v2/checkpoints/resnet56_cifar10_baseline.pth $C10_R56
        echo "Copied from code dir."
    else
        echo "SKIPPING ABLATIONS — no CIFAR-10 baseline found."
    fi
fi

if [ -f "$C10_R56" ]; then
    # === B.1: A7 — Calibration size sensitivity (fast, no training) ===
    echo ""
    echo "============================================================"
    echo "ABLATION A7: Calibration Set Size Sensitivity"
    echo "============================================================"
    python -m experiments.run_ablations \
        --checkpoint $C10_R56 \
        --ablation A7 \
        --model resnet56 \
        --dataset cifar10 \
        --data_path $SCRATCH_DIR/data \
        --save_dir $SCRATCH_DIR/checkpoints \
        --log_dir $SCRATCH_DIR/logs

    # === B.2: A8 — Safety bound tightness (fast, no training) ===
    echo ""
    echo "============================================================"
    echo "ABLATION A8: Safety Bound Tightness"
    echo "============================================================"
    python -m experiments.run_ablations \
        --checkpoint $C10_R56 \
        --ablation A8 \
        --model resnet56 \
        --dataset cifar10 \
        --data_path $SCRATCH_DIR/data \
        --save_dir $SCRATCH_DIR/checkpoints \
        --log_dir $SCRATCH_DIR/logs

    # === B.3: A1 — Block-only vs Block+Channel (requires training) ===
    echo ""
    echo "============================================================"
    echo "ABLATION A1: Block-Only vs Block+Channel"
    echo "============================================================"
    python -m experiments.run_ablations \
        --checkpoint $C10_R56 \
        --ablation A1 \
        --model resnet56 \
        --dataset cifar10 \
        --ft_epochs 200 \
        --data_path $SCRATCH_DIR/data \
        --save_dir $SCRATCH_DIR/checkpoints \
        --log_dir $SCRATCH_DIR/logs

    # === B.4: A5 — Channel criterion comparison (requires training) ===
    echo ""
    echo "============================================================"
    echo "ABLATION A5: Channel Criterion TG vs L1 vs Fisher"
    echo "============================================================"
    python -m experiments.run_ablations \
        --checkpoint $C10_R56 \
        --ablation A5 \
        --model resnet56 \
        --dataset cifar10 \
        --ft_epochs 200 \
        --data_path $SCRATCH_DIR/data \
        --save_dir $SCRATCH_DIR/checkpoints \
        --log_dir $SCRATCH_DIR/logs
fi

echo ""
echo "============================================================"
echo "ALL CIFAR-100 + ABLATION EXPERIMENTS COMPLETED: $(date)"
echo "============================================================"

# Print all summaries
echo ""
echo "=== RESULTS SUMMARY ==="
for f in $SCRATCH_DIR/checkpoints/*cifar100*_summary.json $SCRATCH_DIR/checkpoints/ablation_*.json; do
    if [ -f "$f" ]; then
        echo ""
        echo "--- $(basename $f) ---"
        cat $f | python -c "
import sys, json
d = json.load(sys.stdin)
if isinstance(d, dict) and 'final_acc' in d:
    print(f\"  Acc: {d['final_acc']:.2f}% (delta {d['acc_delta']:+.2f}%)\")
    print(f\"  FLOPs: {d['flops_reduction_pct']:.1f}% reduction\")
    print(f\"  Blocks removed: {d['num_blocks_removed']}\")
elif isinstance(d, list):
    for item in d[:5]:
        print(f\"  {item.get('block','?')}: drop={item.get('actual_acc_drop',0):.2f}%, bound={item.get('lipschitz_bound',0):.6f}\")
else:
    print('  (custom format)')
" 2>/dev/null || echo "  (parse error)"
    fi
done

exit
