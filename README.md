# TransGap v2 — Transformation Gap-Based Multi-Granularity Pruning

## Project Structure

```
transgap_v2/
├── models/
│   ├── __init__.py
│   ├── resnet_cifar.py       # ResNet-56/110 for CIFAR-10/100
│   └── resnet_imagenet.py    # ResNet-50 for ImageNet (torchvision wrapper)
├── metrics/
│   ├── __init__.py
│   ├── transformation_gap.py # Block-level and channel-level TG computation
│   ├── safety_bounds.py      # Lipschitz + DPI safety bound estimation
│   └── flops.py              # FLOPs and parameter counting
├── pruning/
│   ├── __init__.py
│   ├── block_pruner.py       # Block removal via TG
│   ├── channel_pruner.py     # Channel pruning via channel-level TG
│   ├── optimizer.py          # Joint block-channel optimization under FLOPs budget
│   └── surgery.py            # Network structural modification utilities
├── training/
│   ├── __init__.py
│   ├── trainer.py            # Training loop with KD, TG-scaled LR, etc.
│   ├── baseline.py           # Baseline training script
│   └── finetune.py           # Post-pruning fine-tuning with TG-aware compensation
├── experiments/
│   ├── __init__.py
│   ├── run_baseline.py       # Train baseline models
│   ├── run_tg_analysis.py    # Compute and visualize TG scores
│   ├── run_pruning.py        # Full pruning pipeline
│   └── run_ablations.py      # Ablation studies
├── configs/
│   ├── baseline_cifar10.yaml
│   ├── baseline_cifar100.yaml
│   ├── prune_cifar10.yaml
│   └── prune_imagenet.yaml
├── scripts/
│   ├── truba_baseline.slurm  # SLURM: baseline training
│   ├── truba_prune.slurm     # SLURM: pruning pipeline
│   └── truba_ablation.slurm  # SLURM: ablation studies
├── utils/
│   ├── __init__.py
│   ├── logger.py             # Logging utilities
│   ├── checkpoint.py         # Save/load checkpoints
│   └── visualize.py          # TG visualization, comparison plots
├── requirements.txt
└── README.md
```

## Quick Start

```bash
# 1. Train baseline
python -m experiments.run_baseline --config configs/baseline_cifar10.yaml

# 2. Analyze TG scores
python -m experiments.run_tg_analysis --checkpoint checkpoints/resnet56_cifar10_baseline.pth

# 3. Run pruning pipeline
python -m experiments.run_pruning --config configs/prune_cifar10.yaml

# 4. Run ablations
python -m experiments.run_ablations --config configs/prune_cifar10.yaml --ablation A1
```

## On TRUBA

```bash
# Copy to TRUBA
scp -r transgap_v2/ mhassan@172.16.7.1:/arf/home/mhassan/

# Submit baseline training
sbatch scripts/truba_baseline.slurm

# Submit pruning after baseline completes
sbatch scripts/truba_prune.slurm
```
