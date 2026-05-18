#!/bin/bash
#SBATCH -J tg2_fix
#SBATCH -A mhassan
#SBATCH -p akya-cuda
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 10
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/fix_%j.out
#SBATCH --error=logs/fix_%j.err

# Missing experiment: ResNet-110 / CIFAR-100 / 50% FLOPs

SCRATCH_DIR=/arf/scratch/$USER/transgap_v2
CODE_DIR=/arf/home/$USER/transgap_v2

mkdir -p $SCRATCH_DIR/logs
cd $SCRATCH_DIR

module purge
module load comp/python/miniconda3
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

cd $CODE_DIR

C100_R110=$SCRATCH_DIR/checkpoints/resnet110_cifar100_baseline.pth

echo "Running: ResNet-110 / CIFAR-100 / 50% FLOPs"
python -m experiments.run_pruning \
    --checkpoint $C100_R110 \
    --model resnet110 \
    --dataset cifar100 \
    --target_flops 0.5 \
    --max_block_tg 0.05 \
    --ft_epochs 300 \
    --ft_lr 0.01 \
    --data_path $SCRATCH_DIR/data \
    --save_dir $SCRATCH_DIR/checkpoints \
    --log_dir $SCRATCH_DIR/logs

echo "DONE: $(date)"
