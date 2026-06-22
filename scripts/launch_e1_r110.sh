#!/bin/bash
# launch_e1_r110.sh — Submit all E1 block fine-tune jobs in parallel
#
# Workflow:
#   1. (If needed) Run the scan job to discover qualifying blocks:
#        sbatch scripts/truba_e1_scan.slurm
#      Then wait for it to finish, then run this script.
#
#   2. Submit all per-block fine-tune jobs:
#        bash scripts/launch_e1_r110.sh
#
#   3. After all jobs finish, aggregate results:
#        cd /arf/home/mhassan/transgap_v2
#        python -m experiments.run_e1_r110_correlation --aggregate \
#            --save_dir /arf/scratch/mhassan/transgap_v2/checkpoints

set -euo pipefail

SCRATCH=/arf/scratch/mhassan/transgap_v2
SCAN_JSON=$SCRATCH/checkpoints/e1_r110_scan.json
SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"

# ── check scan file exists ──────────────────────────────────────────────────
if [ ! -f "$SCAN_JSON" ]; then
    echo "Scan file not found: $SCAN_JSON"
    echo ""
    echo "Run the scan job first:"
    echo "  sbatch $SCRIPTS_DIR/truba_e1_scan.slurm"
    echo ""
    echo "Then wait for it to complete and re-run this script."
    exit 1
fi

# ── parse qualifying block names from scan JSON ─────────────────────────────
# Uses python one-liner to avoid jq dependency
BLOCKS=$(python3 -c "
import json, sys
with open('$SCAN_JSON') as f:
    d = json.load(f)
for entry in d['qualifying_blocks']:
    print(entry['block'])
")

if [ -z "$BLOCKS" ]; then
    echo "No qualifying blocks found in $SCAN_JSON"
    exit 1
fi

NBLOCKS=$(echo "$BLOCKS" | wc -l | tr -d ' ')
echo "============================================================"
echo "E1: Launching $NBLOCKS per-block fine-tune jobs"
echo "Scan file : $SCAN_JSON"
echo "SLURM job : $SCRIPTS_DIR/truba_e1_r110.slurm"
echo "Time/job  : ~4 h wall, ~1.5–2 h actual"
echo "============================================================"

SUBMITTED=0
for BLOCK in $BLOCKS; do
    JOB_ID=$(sbatch --parsable "$SCRIPTS_DIR/truba_e1_r110.slurm" "$BLOCK")
    echo "  Submitted: $BLOCK  →  job $JOB_ID"
    SUBMITTED=$((SUBMITTED + 1))
done

echo ""
echo "============================================================"
echo "Submitted $SUBMITTED jobs."
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo ""
echo "After all jobs finish, aggregate:"
echo "  cd /arf/home/mhassan/transgap_v2"
echo "  python -m experiments.run_e1_r110_correlation --aggregate \\"
echo "      --save_dir $SCRATCH/checkpoints"
echo "============================================================"
