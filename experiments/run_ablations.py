"""
Run Ablation Studies
=====================
Tests individual contributions of TransGap v2 components.

Ablations:
  A1: Block-only vs Block+Channel (multi-granularity value)
  A2: With vs without safety bounds  
  A5: Channel criterion: TG vs L1-norm vs Fisher
  A7: Calibration set size sensitivity
  A8: Safety bound tightness validation

Usage:
    python -m experiments.run_ablations --checkpoint path/to/baseline.pth \
                                        --ablation A1 --dataset cifar10
"""

import argparse
import os
import sys
import json
import copy
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from models import resnet56, resnet110
from models.resnet_cifar import BasicBlock
from metrics.transformation_gap import compute_block_tg, compute_channel_tg, TGAnalyzer
from metrics.safety_bounds import compute_safety_bounds
from metrics.flops import count_flops, count_params
from pruning.block_pruner import BlockPruner
from pruning.channel_pruner import ChannelPruner
from pruning.optimizer import PruningOptimizer
from training.baseline import get_cifar_loaders
from training.trainer import Trainer
from utils.logger import setup_logger
from utils.checkpoint import load_checkpoint


def load_model_and_data(config, device):
    """Common setup for all ablations."""
    dataset = config.get('dataset', 'cifar10')
    num_classes = 10 if dataset == 'cifar10' else 100
    model_name = config.get('model', 'resnet56')
    
    if model_name == 'resnet56':
        model = resnet56(num_classes=num_classes)
    else:
        model = resnet110(num_classes=num_classes)
    
    checkpoint = load_checkpoint(config['checkpoint'], model, device=device)
    baseline_acc = checkpoint.get('best_acc', 0)
    model.to(device)
    
    train_loader, test_loader, _ = get_cifar_loaders(
        dataset, config.get('batch_size', 128), config.get('data_path', './data')
    )
    
    return model, baseline_acc, train_loader, test_loader, num_classes


def finetune_and_evaluate(model, teacher, train_loader, test_loader, device, config, save_name):
    """Common fine-tuning for all ablations."""
    ft_config = {
        'lr': config.get('ft_lr', 0.01),
        'momentum': 0.9,
        'weight_decay': 5e-4,
        'epochs': config.get('ft_epochs', 200),  # Shorter for ablations
        'warmup_epochs': 5,
        'label_smoothing': 0.1,
        'use_mixup': True,
        'mixup_alpha': 0.2,
        'use_kd': True,
        'kd_alpha': 0.7,
        'kd_temperature': 4.0,
        'grad_clip': 5.0,
    }
    
    save_dir = config.get('save_dir', './checkpoints')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{save_name}.pth")
    
    trainer = Trainer(model, train_loader, test_loader, device, ft_config, teacher=teacher)
    result = trainer.train(save_path=save_path)
    return result


# ============================================================
# A1: Block-only vs Block+Channel
# ============================================================
def run_A1(config):
    """
    A1: Compare block-only pruning vs block+channel pruning at same FLOPs target.
    Shows the value of multi-granularity (Contribution C1).
    """
    print("=" * 60)
    print("ABLATION A1: Block-Only vs Block+Channel")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    target_flops = config.get('target_flops_ratio', 0.5)
    results = {}
    
    for mode in ['block_only', 'block_channel']:
        print(f"\n--- Mode: {mode} ---")
        model, baseline_acc, train_loader, test_loader, _ = load_model_and_data(config, device)
        teacher = copy.deepcopy(model).eval()
        
        # Compute TG
        analyzer = TGAnalyzer(model, device)
        block_tg = analyzer.analyze_blocks(test_loader, num_batches=10)
        
        if mode == 'block_only':
            # Remove blocks aggressively to hit target, no channel pruning
            block_pruner = BlockPruner(model, block_tg)
            blocks = block_pruner.select_blocks_to_remove(max_tg_threshold=0.05)
            
            # Greedily remove until target met
            original_flops = count_flops(model, (1, 3, 32, 32))
            target = int(original_flops * target_flops)
            selected = []
            for name in blocks:
                selected.append(name)
                temp = copy.deepcopy(model)
                for n in selected:
                    temp.replace_block_with_identity(n)
                if count_flops(temp, (1, 3, 32, 32)) <= target:
                    del temp
                    break
                del temp
            
            for name in selected:
                model.replace_block_with_identity(name)
            
            pruned_flops = count_flops(model, (1, 3, 32, 32))
            channel_info = "none"
            
        else:  # block_channel
            # Use the optimizer for balanced block+channel
            channel_tg = analyzer.analyze_channels(test_loader, num_batches=10, method='approx')
            optimizer = PruningOptimizer(model, block_tg, channel_tg, input_size=(1, 3, 32, 32))
            blocks_to_remove, channel_ratios = optimizer.optimize(
                target_flops_ratio=target_flops, max_block_tg=0.02
            )
            
            # Apply block removal
            for name in blocks_to_remove:
                model.replace_block_with_identity(name)
            
            # Apply channel pruning
            if channel_ratios:
                ch_pruner = ChannelPruner(model, channel_tg)
                ch_pruner.prune_all_blocks(channel_ratios)
            
            pruned_flops = count_flops(model, (1, 3, 32, 32))
            channel_info = f"{len(channel_ratios)} blocks"
        
        original_flops = count_flops(teacher, (1, 3, 32, 32))
        flops_red = (1 - pruned_flops / original_flops) * 100
        
        # Fine-tune
        model.to(device)
        result = finetune_and_evaluate(
            model, teacher, train_loader, test_loader, device, config,
            f"ablation_A1_{mode}"
        )
        
        results[mode] = {
            'acc': result['best_acc'],
            'flops_reduction': flops_red,
            'channel_pruning': channel_info,
        }
        
        print(f"  {mode}: {result['best_acc']:.2f}% at {flops_red:.1f}% FLOPs reduction")
    
    print(f"\n--- A1 Summary ---")
    for mode, r in results.items():
        print(f"  {mode}: {r['acc']:.2f}% @ {r['flops_reduction']:.1f}% FLOPs, channels={r['channel_pruning']}")
    
    return results


# ============================================================
# A5: Channel criterion comparison: TG vs L1 vs Fisher
# ============================================================
def run_A5(config):
    """
    A5: Compare channel pruning criteria at same compression level.
    Tests whether channel-level TG outperforms standard alternatives.
    """
    print("=" * 60)
    print("ABLATION A5: Channel Criterion — TG vs L1 vs Fisher")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = {}
    
    for criterion in ['tg', 'l1', 'fisher']:
        print(f"\n--- Criterion: {criterion} ---")
        model, baseline_acc, train_loader, test_loader, _ = load_model_and_data(config, device)
        teacher = copy.deepcopy(model).eval()
        
        # Compute block TG and remove some blocks first
        analyzer = TGAnalyzer(model, device)
        block_tg = analyzer.analyze_blocks(test_loader, num_batches=10)
        
        # Remove 7 laziest blocks (same for all criteria)
        block_pruner = BlockPruner(model, block_tg)
        blocks = block_pruner.select_blocks_to_remove(num_blocks=7, max_tg_threshold=0.05)
        for name in blocks:
            model.replace_block_with_identity(name)
        
        # Now prune channels with different criteria
        prune_ratio = 0.3  # Same ratio for all
        surviving_blocks = [
            name for name, tg in block_tg.items()
            if tg == tg and name not in blocks  # not NaN and not removed
        ]
        
        for block_name in surviving_blocks:
            layer_name, block_idx = block_name.rsplit('.', 1)
            layer = getattr(model, layer_name)
            block = layer[int(block_idx)]
            
            if not isinstance(block, BasicBlock):
                continue
            
            num_ch = block.conv2.weight.shape[0]
            num_keep = max(1, int(num_ch * (1 - prune_ratio)))
            
            if criterion == 'tg':
                # Use channel-level TG
                ctg = compute_channel_tg(model, test_loader, device, block_name, num_batches=10, method='approx')
                keep_idx = torch.argsort(ctg, descending=True)[:num_keep].sort().values
            elif criterion == 'l1':
                # L1-norm of conv2 weights
                l1_scores = block.conv2.weight.data.abs().sum(dim=(1, 2, 3))
                keep_idx = torch.argsort(l1_scores, descending=True)[:num_keep].sort().values
            elif criterion == 'fisher':
                # Fisher information approximation: gradient² × activation²
                # Simplified: use weight magnitude × BN scaling
                bn_scale = block.bn2.weight.data.abs()
                w_norm = block.conv2.weight.data.pow(2).sum(dim=(1, 2, 3)).sqrt()
                fisher_scores = bn_scale * w_norm
                keep_idx = torch.argsort(fisher_scores, descending=True)[:num_keep].sort().values
            
            # Prune conv1 outputs and conv2 inputs
            from pruning.channel_pruner import prune_conv_layer
            block.conv1, block.bn1 = prune_conv_layer(block.conv1, block.bn1, keep_idx, dim='output')
            block.conv2, block.bn2 = prune_conv_layer(block.conv2, block.bn2, keep_idx, dim='input')
        
        original_flops = count_flops(teacher, (1, 3, 32, 32))
        pruned_flops = count_flops(model, (1, 3, 32, 32))
        flops_red = (1 - pruned_flops / original_flops) * 100
        
        # Fine-tune
        model.to(device)
        result = finetune_and_evaluate(
            model, teacher, train_loader, test_loader, device, config,
            f"ablation_A5_{criterion}"
        )
        
        results[criterion] = {
            'acc': result['best_acc'],
            'flops_reduction': flops_red,
        }
        print(f"  {criterion}: {result['best_acc']:.2f}% at {flops_red:.1f}% FLOPs reduction")
    
    print(f"\n--- A5 Summary ---")
    for crit, r in results.items():
        print(f"  {crit}: {r['acc']:.2f}% @ {r['flops_reduction']:.1f}% FLOPs")
    
    return results


# ============================================================
# A7: Calibration set size sensitivity
# ============================================================
def run_A7(config):
    """
    A7: Test TG stability across different calibration set sizes.
    Shows that TG scores are robust and don't require large calibration sets.
    """
    print("=" * 60)
    print("ABLATION A7: Calibration Set Size Sensitivity")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, baseline_acc, train_loader, test_loader, _ = load_model_and_data(config, device)
    
    # Test with different numbers of calibration batches
    # batch_size=128, so 1 batch=128, 4=512, 8=1024, 40=5120
    cal_sizes = [1, 2, 4, 8, 20, 40]
    results = {}
    
    for n_batches in cal_sizes:
        n_samples = n_batches * 128
        print(f"\n--- Calibration: {n_batches} batches ({n_samples} samples) ---")
        
        block_tg = compute_block_tg(model, test_loader, device, num_batches=n_batches)
        
        # Record TG values
        tg_values = {k: v for k, v in block_tg.items() if v == v}
        results[n_samples] = tg_values
        
        # Print top-5 laziest
        ranked = sorted(tg_values.items(), key=lambda x: x[1])
        print(f"  Top-5 laziest: {[(n, f'{tg:.6f}') for n, tg in ranked[:5]]}")
    
    # Compute stability: correlation between different calibration sizes
    import numpy as np
    baseline_tg = results[max(results.keys())]  # largest calibration as reference
    block_names = sorted(baseline_tg.keys())
    
    print(f"\n--- A7 Stability Analysis ---")
    print(f"{'Cal. Size':<12} {'Rank Corr.':<14} {'Max TG Diff':<14} {'Block Order Stable'}")
    
    ref_values = np.array([baseline_tg[n] for n in block_names])
    ref_order = np.argsort(ref_values)
    
    for n_samples, tg_dict in sorted(results.items()):
        values = np.array([tg_dict[n] for n in block_names])
        order = np.argsort(values)
        
        # Spearman rank correlation
        try:
            from scipy.stats import spearmanr
            corr, _ = spearmanr(ref_values, values)
        except ImportError:
            # Fallback: manual Spearman via rank correlation
            def _rank(arr):
                temp = arr.argsort().argsort().astype(float)
                return temp
            corr = np.corrcoef(_rank(ref_values), _rank(values))[0, 1]
        
        max_diff = np.max(np.abs(ref_values - values))
        order_stable = "Yes" if np.array_equal(order[:7], ref_order[:7]) else "Partial"
        
        print(f"{n_samples:<12} {corr:<14.4f} {max_diff:<14.6f} {order_stable}")
    
    # Save results
    save_dir = config.get('save_dir', './checkpoints')
    os.makedirs(save_dir, exist_ok=True)
    summary_path = os.path.join(save_dir, "ablation_A7_summary.json")
    with open(summary_path, 'w') as f:
        json.dump({k: {n: round(v, 8) for n, v in tg.items()} for k, tg in results.items()}, f, indent=2)
    print(f"\nSaved: {summary_path}")
    
    return results


# ============================================================
# A8: Safety bound tightness validation
# ============================================================
def run_A8(config):
    """
    A8: Compare predicted safety bounds with actual accuracy drops.
    Validates Theorems 1 and 2.
    """
    print("=" * 60)
    print("ABLATION A8: Safety Bound Tightness Validation")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, baseline_acc, train_loader, test_loader, _ = load_model_and_data(config, device)
    teacher = copy.deepcopy(model).eval()
    
    # Compute TG and safety bounds
    analyzer = TGAnalyzer(model, device)
    block_tg = analyzer.analyze_blocks(test_loader, num_batches=10)
    safety_bounds = compute_safety_bounds(model, block_tg, test_loader, device, num_batches=10)
    
    # Test: remove each block individually, measure actual accuracy drop
    prunable = [(n, tg) for n, tg in block_tg.items() if tg == tg]
    prunable.sort(key=lambda x: x[1])
    
    # Test the 10 laziest blocks individually
    results = []
    
    for name, tg in prunable[:10]:
        print(f"\n  Testing removal of {name} (TG={tg:.6f})...")
        test_model = copy.deepcopy(model)
        test_model.replace_block_with_identity(name)
        test_model.to(device)
        
        # Evaluate without fine-tuning
        temp_trainer = Trainer(test_model, train_loader, test_loader, device, {})
        acc_after = temp_trainer.evaluate()
        acc_drop = baseline_acc - acc_after
        
        predicted = safety_bounds[name]
        
        results.append({
            'block': name,
            'tg': tg,
            'actual_acc_drop': acc_drop,
            'lipschitz_bound': predicted['lipschitz_bound'],
            'dpi_info_loss': predicted['dpi_info_loss'],
        })
        
        print(f"    Actual drop: {acc_drop:.2f}%, Lipschitz: {predicted['lipschitz_bound']:.6f}, DPI: {predicted['dpi_info_loss']:.4f}")
    
    # Summary
    print(f"\n--- A8 Summary ---")
    print(f"{'Block':<15} {'TG':>10} {'Actual Drop':>12} {'Lip. Bound':>12} {'DPI Loss':>10}")
    print("-" * 60)
    for r in results:
        print(f"{r['block']:<15} {r['tg']:>10.6f} {r['actual_acc_drop']:>11.2f}% {r['lipschitz_bound']:>12.6f} {r['dpi_info_loss']:>10.4f}")
    
    # Save
    save_dir = config.get('save_dir', './checkpoints')
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "ablation_A8_summary.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


# ============================================================
# MAIN
# ============================================================
ABLATION_MAP = {
    'A1': run_A1,
    'A5': run_A5,
    'A7': run_A7,
    'A8': run_A8,
}

def main():
    parser = argparse.ArgumentParser(description='TransGap v2 — Ablation Studies')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--ablation', type=str, required=True, choices=list(ABLATION_MAP.keys()) + ['all'])
    parser.add_argument('--model', type=str, default='resnet56')
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--target_flops', type=float, default=0.5)
    parser.add_argument('--ft_epochs', type=int, default=200)
    parser.add_argument('--ft_lr', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--log_dir', type=str, default='./logs')
    args = parser.parse_args()
    
    config = {
        'model': args.model,
        'dataset': args.dataset,
        'checkpoint': args.checkpoint,
        'target_flops_ratio': args.target_flops,
        'ft_epochs': args.ft_epochs,
        'ft_lr': args.ft_lr,
        'batch_size': args.batch_size,
        'data_path': args.data_path,
        'save_dir': args.save_dir,
    }
    
    setup_logger(args.log_dir, f"ablation_{args.ablation}")
    
    if args.ablation == 'all':
        for name, fn in ABLATION_MAP.items():
            print(f"\n\n{'#' * 60}")
            print(f"# Running {name}")
            print(f"{'#' * 60}")
            fn(config)
    else:
        ABLATION_MAP[args.ablation](config)


if __name__ == '__main__':
    main()
