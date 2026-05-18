#!/bin/bash
# =============================================================================
# launch_phase2.sh — TransGap v2 Phase 2 Revision Master Launcher
# =============================================================================
#
# USAGE:
#   cd /arf/scratch/mhassan/transgap_v2
#   bash /arf/home/mhassan/transgap_v2/scripts/launch_phase2.sh [options]
#
# OPTIONS:
#   --dry-run          Print commands without submitting
#   --only <group>     Submit only one group:
#                        fast      Items 10+11 (~2h, no training)
#                        geometry  Items 7+12  (~2h, no training)
#                        ablation  Items 8+9   (~16-48h, training)
#                        seeds     Item 6      (~12h per job, training)
#                        all       Everything [default]
#
# EXAMPLES:
#   bash launch_phase2.sh                    # submit everything
#   bash launch_phase2.sh --dry-run          # preview only
#   bash launch_phase2.sh --only fast        # just Items 10+11
#   bash launch_phase2.sh --only seeds       # just multi-seed runs
#
# RUN FROM: /arf/scratch/mhassan/transgap_v2  (not home dir)
# =============================================================================

set -euo pipefail

SCRIPT_DIR=/arf/home/mhassan/transgap_v2/scripts
SCRATCH=/arf/scratch/mhassan/transgap_v2
DRY_RUN=0
ONLY="all"

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run) DRY_RUN=1; shift ;;
        --only) ONLY=$2; shift 2 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

mkdir -p $SCRATCH/logs $SCRATCH/checkpoints

submit() {
    local label=$1; shift
    if [ $DRY_RUN -eq 1 ]; then
        echo "[DRY-RUN] sbatch $@ ($label)"
    else
        JID=$(sbatch "$@" | awk '{print $NF}')
        echo "  Submitted $label → Job $JID"
    fi
}

echo "============================================================"
echo "TransGap v2 — Phase 2 Revision Launch"
echo "Mode: $([ $DRY_RUN -eq 1 ] && echo DRY-RUN || echo LIVE)"
echo "Group: $ONLY"
echo "Time: $(date)"
echo "============================================================"

# ── Items 10+11: fast, no training ───────────────────────────────────────────
if [[ "$ONLY" == "all" || "$ONLY" == "fast" ]]; then
    echo ""
    echo "--- Items 10+11: Channel approx vs exact + Calibration subsets (~2h) ---"
    submit "items10+11" $SCRIPT_DIR/truba_revision_fast.slurm
fi

# ── Items 7+12: orthogonality, no training ───────────────────────────────────
if [[ "$ONLY" == "all" || "$ONLY" == "geometry" ]]; then
    echo ""
    echo "--- Items 7+12: Orthogonality validation (~2h) ---"
    submit "orthog_cifar" $SCRIPT_DIR/truba_revision_orthog.slurm r56_cifar10
    submit "orthog_r110"  $SCRIPT_DIR/truba_revision_orthog.slurm r110_cifar10
    submit "orthog_imgnet" $SCRIPT_DIR/truba_revision_orthog.slurm r50_imagenet
fi

# ── Items 8+9: criterion + recipe ablation ───────────────────────────────────
if [[ "$ONLY" == "all" || "$ONLY" == "ablation" ]]; then
    echo ""
    echo "--- Item 8: TG vs Residual Norm (3 seeds × ~16h each) ---"
    for SEED in 42 123 456; do
        submit "resnorm_seed${SEED}" $SCRIPT_DIR/truba_revision_resnorm.slurm $SEED
    done

    echo ""
    echo "--- Item 9: Recipe ablation (4 conditions × ~12h, all in one job) ---"
    submit "recipe_all" $SCRIPT_DIR/truba_revision_recipe.slurm all
fi

# ── Item 6: multi-seed fine-tuning ───────────────────────────────────────────
if [[ "$ONLY" == "all" || "$ONLY" == "seeds" ]]; then
    echo ""
    echo "--- Item 6: Multi-seed CIFAR (3 settings × 3 seeds = 9 jobs) ---"
    for SETTING in r56_cifar10_f05 r56_cifar10_f06 r110_cifar10_f06; do
        for SEED in 42 123 456; do
            submit "seed_${SETTING}_${SEED}" \
                $SCRIPT_DIR/truba_revision_multiseed.slurm $SETTING $SEED
        done
    done

    echo ""
    echo "--- Item 6: Multi-seed ImageNet (2 settings × 3 seeds = 6 jobs) ---"
    echo "    Note: re-fine-tunes from existing pruned_state.pth checkpoints"
    for SETTING in r50_imagenet_f06 r50_imagenet_f05; do
        for SEED in 42 123 456; do
            submit "seed_${SETTING}_${SEED}" \
                $SCRIPT_DIR/truba_revision_multiseed.slurm $SETTING $SEED
        done
    done
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "Done. Monitor: squeue -u mhassan"
echo ""
echo "Logs: $SCRATCH/logs/"
echo "Results: $SCRATCH/checkpoints/"
echo ""
echo "After all jobs complete, aggregate:"
echo "  cd /arf/home/mhassan/transgap_v2"
echo "  python -m experiments.run_multiseed --setting aggregate_all --aggregate --save_dir $SCRATCH/checkpoints"
echo "  python -m experiments.run_ablation_residual_norm --aggregate --save_dir $SCRATCH/checkpoints"
echo "  python -m experiments.run_ablation_recipe --condition aggregate --save_dir $SCRATCH/checkpoints"
echo "============================================================"
