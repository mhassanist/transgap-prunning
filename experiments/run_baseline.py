"""
Run Baseline Training
======================
Train ResNet-56/110 from scratch to establish baseline accuracy.
Target: 94.30%+ for ResNet-56 on CIFAR-10.

Usage:
    python -m experiments.run_baseline --model resnet56 --dataset cifar10
    python -m experiments.run_baseline --config configs/baseline_cifar10.yaml
"""

import argparse
import yaml
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.baseline import train_baseline
from utils.logger import setup_logger


def main():
    parser = argparse.ArgumentParser(description='TransGap v2 — Baseline Training')
    parser.add_argument('--config', type=str, default=None, help='Path to YAML config')
    parser.add_argument('--model', type=str, default='resnet56', choices=['resnet56', 'resnet110'])
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100'])
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--log_dir', type=str, default='./logs')
    parser.add_argument('--resume', action='store_true', help='Resume from latest checkpoint')
    args = parser.parse_args()
    
    # Load config from YAML or build from args
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = {
            'model': args.model,
            'dataset': args.dataset,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'data_path': args.data_path,
            'save_dir': args.save_dir,
        }
    
    # CLI paths always override YAML (critical for TRUBA scratch dir)
    config['save_dir'] = args.save_dir
    config['data_path'] = args.data_path
    if args.model != 'resnet56':
        config['model'] = args.model
    if args.dataset != 'cifar10':
        config['dataset'] = args.dataset
    
    # Setup logging
    setup_logger(args.log_dir, f"baseline_{config['model']}_{config['dataset']}")
    
    print("=" * 60)
    print("TransGap v2 — Baseline Training")
    print("=" * 60)
    print(f"Config: {config}")
    print()
    
    result = train_baseline(config)
    
    print(f"\nFinal Result: {result['best_acc']:.2f}%")


if __name__ == '__main__':
    main()
