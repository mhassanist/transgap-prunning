# TransGap v2 — Transformation Gap-Based Multi-Granularity Pruning

Structured pruning via **Transformation Gap (TG)**: a data-driven score that measures
how much each residual block or channel changes its input. Low-TG units are functionally
redundant and safe to remove.

## Repository structure

```
transgap_v2/
├── models/
│   ├── resnet_cifar.py         # ResNet-56/110 for CIFAR-10/100 (BasicBlock, IdentityBlock)
│   └── resnet_imagenet.py      # ResNet-50 for ImageNet (torchvision wrapper)
├── metrics/
│   ├── transformation_gap.py   # Block-level and channel-level TG (hooks, TGAnalyzer)
│   ├── safety_bounds.py        # Lipschitz / DPI safety bound estimation
│   └── flops.py                # FLOPs and parameter counting
├── pruning/
│   ├── block_pruner.py         # Remove blocks below TG threshold → IdentityBlock
│   ├── channel_pruner.py       # Channel pruning ranked by channel-level TG
│   └── optimizer.py            # Joint block+channel selection under FLOPs budget
├── training/
│   ├── trainer.py              # Unified loop: CE + KD + Mixup + cosine LR + TG-scaled LR
│   └── baseline.py             # Train from scratch; provides get_cifar_loaders()
├── experiments/
│   ├── run_baseline.py         # Train ResNet-56/110 baselines
│   ├── run_pruning.py          # Full pruning pipeline (TG → block+channel → finetune)
│   ├── run_ablations.py        # Ablation suite (A1–A7)
│   ├── run_ablation_A8_full.py # A8: single-block removal vs TG (R56/R110/R50)
│   ├── run_e1_r110_correlation.py  # E1: cross-backbone TG correlation (R110/CIFAR-10)
│   ├── run_orthogonality.py    # Orthogonality / TG distribution analysis
│   ├── run_multiseed.py        # Multi-seed robustness runs
│   ├── run_imagenet.py         # ImageNet pruning (ResNet-50)
│   ├── run_imagenet_full.py    # ImageNet full pipeline
│   ├── run_ablation_recipe.py  # Fine-tune recipe ablation
│   ├── run_ablation_channel_exact.py  # Exact vs approximate channel TG
│   └── run_ablation_residual_norm.py  # Residual norm ablation
├── configs/
│   ├── baseline_cifar10.yaml       # ResNet-56 / CIFAR-10 baseline
│   ├── baseline_cifar10_r110.yaml  # ResNet-110 / CIFAR-10 baseline
│   ├── prune_cifar10.yaml          # ResNet-56 pruning config
│   └── prune_cifar10_r110.yaml     # ResNet-110 pruning config
├── scripts/                    # SLURM job scripts for Truba HPC
│   ├── truba_baseline.slurm
│   ├── truba_prune.slurm
│   ├── truba_r110.slurm
│   ├── truba_ablation_A8.slurm
│   ├── truba_e1_scan.slurm     # E1: TG scan (run once, ~20 min)
│   ├── truba_e1_r110.slurm     # E1: single-block fine-tune (~2 h each)
│   ├── launch_e1_r110.sh       # E1: parallel launcher (reads scan JSON)
│   └── ...
└── utils/
    ├── checkpoint.py           # save_checkpoint / load_checkpoint
    └── logger.py               # File + console logging
```

## Quick start

```bash
pip install -r requirements.txt

# Train baseline
python -m experiments.run_baseline --model resnet56 --config configs/baseline_cifar10.yaml \
    --data_path ./data --save_dir ./checkpoints

# Full pruning pipeline
python -m experiments.run_pruning \
    --checkpoint checkpoints/resnet56_cifar10_baseline.pth \
    --model resnet56 --dataset cifar10 \
    --target_flops 0.5 --max_block_tg 0.025 \
    --data_path ./data --save_dir ./checkpoints
```

## E1 — Cross-backbone TG correlation (ResNet-110 / CIFAR-10)

Validates that TG rank predicts empirical accuracy drop across backbones.
Removes each low-TG block individually, fine-tunes 150 epochs (KD + Mixup + cosine LR),
and computes Pearson r between TG rank and Δacc.

```bash
# Step 1 — discover qualifying blocks (TG < 0.025), ~20 min
sbatch scripts/truba_e1_scan.slurm

# Step 2 — launch all per-block fine-tune jobs in parallel
bash scripts/launch_e1_r110.sh

# Step 3 — aggregate: compute Pearson r
python -m experiments.run_e1_r110_correlation --aggregate \
    --save_dir /arf/scratch/mhassan/transgap_v2/checkpoints
```

## On Truba HPC

```bash
# Sync code (checkpoints and data stay on scratch)
rsync -av --exclude='.git' --exclude='*.pth' --exclude='data/' --exclude='__pycache__' \
    . mhassan@172.16.6.11:/arf/home/mhassan/transgap_v2/

# Submit from scratch (Truba policy requirement)
cd /arf/scratch/mhassan/transgap_v2
sbatch /arf/home/mhassan/transgap_v2/scripts/truba_baseline.slurm
```

Key paths on cluster:
- Code: `/arf/home/mhassan/transgap_v2/`
- Data + checkpoints + logs: `/arf/scratch/mhassan/transgap_v2/`
