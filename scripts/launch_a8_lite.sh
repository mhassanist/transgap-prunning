#!/bin/bash
# A8 lite sweep — n=15 total cross-backbone single-block removals
cd /arf/scratch/mhassan/transgap_v2
SLURM=/arf/home/mhassan/transgap_v2/scripts/truba_ablation_A8.slurm

# R56 — 6 lowest-TG blocks (5 new + test job 5782851 already covers layer1.8)
for B in layer2.6 layer2.8 layer1.7 layer2.5 layer1.5 layer1.3; do
    sbatch $SLURM r56_cifar10 $B
done

# R110 — 5 lowest-TG blocks (mix of low + medium TG for spread)
for B in layer2.14 layer1.5 layer2.5 layer1.9 layer3.8; do
    sbatch $SLURM r110_cifar10 $B
done

# R50-ImageNet — 3 blocks (smallest TG values, also spans layer1/2/3)
for B in layer1.1 layer2.3 layer3.3; do
    sbatch $SLURM r50_imagenet $B
done

squeue -u mhassan | head -20
