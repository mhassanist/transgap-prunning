#!/bin/bash
cd /arf/scratch/mhassan/transgap_v2
# Submit all 30 A8 single-block jobs across three backbones.

SLURM=/arf/home/mhassan/transgap_v2/scripts/truba_ablation_A8.slurm

# R56 — 10 blocks
for B in layer1.8 layer2.6 layer2.8 layer1.7 layer2.5 layer2.4 layer2.7 layer1.5 layer1.3 layer2.3; do
    sbatch $SLURM r56_cifar10 $B
done

# R110 — 12 blocks
for B in layer2.12 layer2.13 layer2.14 layer2.15 layer2.16 layer2.17 layer1.5 layer2.5 layer1.9 layer1.6 layer1.14 layer2.10; do
    sbatch $SLURM r110_cifar10 $B
done

# R50-ImageNet — 8 blocks
for B in layer1.1 layer1.2 layer2.3 layer2.2 layer3.3 layer3.2 layer3.4 layer2.1; do
    sbatch $SLURM r50_imagenet $B
done

squeue -u mhassan | head
