"""
Run Full Pruning Pipeline
===========================
TransGap v2: Block removal + Channel pruning + Fine-tuning

Pipeline:
  1. Load pretrained baseline
  2. Compute block-level TG scores
  3. Compute channel-level TG scores
  4. Estimate safety bounds
  5. Optimize: select blocks + allocate channel ratios
  6. Execute pruning
  7. Fine-tune with KD + TG-scaled LR

Usage:
    python -m experiments.run_pruning --checkpoint checkpoints/resnet56_cifar10_baseline.pth \
                                      --target_flops 0.5 --dataset cifar10
"""

import argparse
import yaml
import os
import sys
import json
import time
import copy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from models import resnet56, resnet110
from models.resnet_cifar import BasicBlock
from metrics.transformation_gap import TGAnalyzer
from metrics.safety_bounds import compute_safety_bounds
from metrics.flops import count_flops, count_params, model_summary
from pruning.block_pruner import BlockPruner
from pruning.channel_pruner import ChannelPruner
from pruning.optimizer import PruningOptimizer
from training.baseline import get_cifar_loaders
from training.trainer import Trainer
from utils.logger import setup_logger
from utils.checkpoint import load_checkpoint


def run_pruning(config: dict):
    """Full TransGap v2 pruning pipeline."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # ============================
    # Stage 0: Load baseline model
    # ============================
    print("\n" + "=" * 60)
    print("Stage 0: Loading Baseline Model")
    print("=" * 60)
    
    dataset = config.get('dataset', 'cifar10')
    num_classes = 10 if dataset == 'cifar10' else 100
    model_name = config.get('model', 'resnet56')
    
    if model_name == 'resnet56':
        model = resnet56(num_classes=num_classes)
    elif model_name == 'resnet110':
        model = resnet110(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Load checkpoint
    checkpoint_path = config['checkpoint']
    checkpoint = load_checkpoint(checkpoint_path, model, device=device)
    baseline_acc = checkpoint.get('best_acc', 0)
    print(f"Loaded: {checkpoint_path}")
    print(f"Baseline accuracy: {baseline_acc:.2f}%")
    
    # Keep teacher copy for KD
    teacher = copy.deepcopy(model)
    teacher.eval()
    
    model.to(device)
    teacher.to(device)
    
    # Data
    data_path = config.get('data_path', './data')
    batch_size = config.get('batch_size', 128)
    train_loader, test_loader, _ = get_cifar_loaders(dataset, batch_size, data_path)
    
    # Baseline stats
    original_flops = count_flops(model, (1, 3, 32, 32))
    original_params = count_params(model)
    print(f"Original FLOPs: {original_flops:,} ({original_flops/1e6:.2f}M)")
    print(f"Original Params: {original_params:,} ({original_params/1e6:.2f}M)")
    
    # ============================
    # Stage 1: TG Computation
    # ============================
    print("\n" + "=" * 60)
    print("Stage 1: Computing Transformation Gaps")
    print("=" * 60)
    
    num_cal_batches = config.get('num_cal_batches', 10)
    channel_tg_method = config.get('channel_tg_method', 'approx')
    
    analyzer = TGAnalyzer(model, device)
    
    # Block-level TG
    print("\nComputing block-level TG...")
    block_tg = analyzer.analyze_blocks(test_loader, num_batches=num_cal_batches)
    
    # Channel-level TG (for surviving blocks only — we'll analyze all for now)
    print("\nComputing channel-level TG...")
    prunable_names = [n for n, tg in block_tg.items() if tg == tg]  # exclude NaN
    channel_tg = analyzer.analyze_channels(
        test_loader, block_names=prunable_names,
        num_batches=num_cal_batches, method=channel_tg_method,
    )
    
    print(analyzer.summary())
    
    # ============================
    # Stage 2: Safety Bounds
    # ============================
    print("\n" + "=" * 60)
    print("Stage 2: Estimating Safety Bounds")
    print("=" * 60)
    
    safety_bounds = compute_safety_bounds(
        model, block_tg, test_loader, device, num_batches=num_cal_batches
    )
    
    print(f"\n{'Block':<15} {'TG':>10} {'Lipschitz':>12} {'DPI Loss':>12}")
    print("-" * 50)
    for name in sorted(safety_bounds.keys()):
        b = safety_bounds[name]
        print(f"{name:<15} {b['tg']:>10.6f} {b['lipschitz_bound']:>12.4f} {b['dpi_info_loss']:>12.4f}")
    
    # ============================
    # Stage 3: Optimization
    # ============================
    print("\n" + "=" * 60)
    print("Stage 3: Joint Block-Channel Optimization")
    print("=" * 60)
    
    target_flops = config.get('target_flops_ratio', 0.5)
    max_block_tg = config.get('max_block_tg', 0.05)
    max_channel_ratio = config.get('max_channel_ratio', 0.7)
    
    optimizer = PruningOptimizer(
        model, block_tg, channel_tg, safety_bounds, input_size=(1, 3, 32, 32)
    )
    
    blocks_to_remove, channel_ratios = optimizer.optimize(
        target_flops_ratio=target_flops,
        max_block_tg=max_block_tg,
        max_channel_ratio=max_channel_ratio,
    )
    
    print(optimizer.summary(blocks_to_remove, channel_ratios))
    
    # ============================
    # Stage 4: Execute Pruning
    # ============================
    print("\n" + "=" * 60)
    print("Stage 4: Executing Structural Pruning")
    print("=" * 60)
    
    # Block removal
    if blocks_to_remove:
        block_pruner = BlockPruner(model, block_tg, safety_bounds)
        model, removed = block_pruner.prune(num_blocks=len(blocks_to_remove))
    
    # Channel pruning
    if channel_ratios:
        channel_pruner = ChannelPruner(model, channel_tg)
        channel_pruner.prune_all_blocks(channel_ratios)
    
    # Post-pruning stats
    pruned_flops = count_flops(model, (1, 3, 32, 32))
    pruned_params = count_params(model)
    print(f"\nPruned FLOPs: {pruned_flops:,} ({pruned_flops/1e6:.2f}M)")
    print(f"Pruned Params: {pruned_params:,} ({pruned_params/1e6:.2f}M)")
    print(f"FLOPs reduction: {(1 - pruned_flops/original_flops)*100:.1f}%")
    print(f"Params reduction: {(1 - pruned_params/original_params)*100:.1f}%")
    
    # Accuracy before fine-tuning
    model.to(device)
    pre_ft_trainer = Trainer(model, train_loader, test_loader, device, {})
    pre_ft_acc = pre_ft_trainer.evaluate()
    print(f"Accuracy before fine-tuning: {pre_ft_acc:.2f}%")
    
    # ============================
    # Stage 5: Fine-tuning
    # ============================
    print("\n" + "=" * 60)
    print("Stage 5: TG-Aware Fine-tuning")
    print("=" * 60)
    
    # Build TG-scaled learning rates
    tg_lr_scale = {}
    if blocks_to_remove:
        block_tg_values = {n: block_tg.get(n, 0) for n in blocks_to_remove}
        all_block_names = model.block_names
        
        # Scale up LR for blocks adjacent to removed ones
        alpha = config.get('tg_lr_alpha', 1.0)
        for name in all_block_names:
            layer_name, idx_str = name.rsplit('.', 1)
            idx = int(idx_str)
            
            # Check if any neighbor was removed
            neighbor_tg_sum = 0
            for offset in [-2, -1, 1, 2]:
                neighbor_name = f"{layer_name}.{idx + offset}"
                if neighbor_name in blocks_to_remove:
                    neighbor_tg_sum += block_tg.get(neighbor_name, 0)
            
            if neighbor_tg_sum > 0:
                scale = 1.0 + alpha * neighbor_tg_sum
                tg_lr_scale[name] = scale
    
    # Fine-tuning config
    ft_config = {
        'lr': config.get('ft_lr', 0.01),
        'momentum': 0.9,
        'weight_decay': config.get('ft_weight_decay', 5e-4),
        'epochs': config.get('ft_epochs', 300),
        'warmup_epochs': config.get('ft_warmup_epochs', 5),
        'label_smoothing': config.get('label_smoothing', 0.1),
        'use_mixup': config.get('use_mixup', True),
        'mixup_alpha': config.get('mixup_alpha', 0.2),
        'use_kd': True,
        'kd_alpha': config.get('kd_alpha', 0.7),
        'kd_temperature': config.get('kd_temperature', 4.0),
        'grad_clip': config.get('grad_clip', 5.0),
        'tg_lr_scale': tg_lr_scale if tg_lr_scale else None,
    }
    
    save_dir = config.get('save_dir', './checkpoints')
    os.makedirs(save_dir, exist_ok=True)
    
    exp_name = f"{model_name}_{dataset}_pruned_b{len(blocks_to_remove)}_f{target_flops}"
    save_path = os.path.join(save_dir, f"{exp_name}.pth")
    
    trainer = Trainer(model, train_loader, test_loader, device, ft_config, teacher=teacher)
    result = trainer.train(save_path=save_path)
    
    # ============================
    # Final Summary
    # ============================
    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)
    
    summary = {
        'model': model_name,
        'dataset': dataset,
        'baseline_acc': baseline_acc,
        'pre_finetune_acc': pre_ft_acc,
        'final_acc': result['best_acc'],
        'acc_delta': result['best_acc'] - baseline_acc,
        'original_flops': original_flops,
        'pruned_flops': pruned_flops,
        'flops_reduction_pct': (1 - pruned_flops / original_flops) * 100,
        'original_params': original_params,
        'pruned_params': pruned_params,
        'params_reduction_pct': (1 - pruned_params / original_params) * 100,
        'blocks_removed': blocks_to_remove,
        'num_blocks_removed': len(blocks_to_remove),
        'channel_ratios': {k: round(v, 3) for k, v in channel_ratios.items()},
        'block_tg': {k: round(v, 6) if v == v else 'NaN' for k, v in block_tg.items()},
        'training_time_hours': result['total_time'] / 3600,
    }
    
    print(f"Baseline Accuracy:    {baseline_acc:.2f}%")
    print(f"Before Fine-tuning:   {pre_ft_acc:.2f}%")
    print(f"Final Accuracy:       {result['best_acc']:.2f}%")
    print(f"Accuracy Delta:       {result['best_acc'] - baseline_acc:+.2f}%")
    print(f"FLOPs Reduction:      {summary['flops_reduction_pct']:.1f}%")
    print(f"Params Reduction:     {summary['params_reduction_pct']:.1f}%")
    print(f"Blocks Removed:       {len(blocks_to_remove)}")
    print(f"Training Time:        {result['total_time']/3600:.2f}h")
    
    # Save summary
    summary_path = os.path.join(save_dir, f"{exp_name}_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved: {summary_path}")
    
    return summary


def main():
    parser = argparse.ArgumentParser(description='TransGap v2 — Full Pruning Pipeline')
    parser.add_argument('--config', type=str, default=None, help='YAML config path')
    parser.add_argument('--checkpoint', type=str, required=True, help='Baseline checkpoint')
    parser.add_argument('--model', type=str, default='resnet56')
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--target_flops', type=float, default=0.5, help='Target FLOPs ratio')
    parser.add_argument('--max_block_tg', type=float, default=0.05)
    parser.add_argument('--ft_epochs', type=int, default=300)
    parser.add_argument('--ft_lr', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--log_dir', type=str, default='./logs')
    args = parser.parse_args()
    
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        config['checkpoint'] = args.checkpoint
    else:
        config = {
            'model': args.model,
            'dataset': args.dataset,
            'checkpoint': args.checkpoint,
            'target_flops_ratio': args.target_flops,
            'max_block_tg': args.max_block_tg,
            'ft_epochs': args.ft_epochs,
            'ft_lr': args.ft_lr,
            'batch_size': args.batch_size,
            'data_path': args.data_path,
            'save_dir': args.save_dir,
        }
    
    # CLI args always override YAML for paths and key params
    config['save_dir'] = args.save_dir
    config['data_path'] = args.data_path
    config['model'] = args.model
    config['dataset'] = args.dataset
    config['target_flops_ratio'] = args.target_flops
    config['max_block_tg'] = args.max_block_tg
    config['ft_epochs'] = args.ft_epochs
    config['ft_lr'] = args.ft_lr
    
    setup_logger(args.log_dir, f"prune_{config['model']}_{config['dataset']}")
    
    print("=" * 60)
    print("TransGap v2 — Full Pruning Pipeline")
    print("=" * 60)
    print(f"Config: {json.dumps(config, indent=2)}")
    
    run_pruning(config)


if __name__ == '__main__':
    main()
