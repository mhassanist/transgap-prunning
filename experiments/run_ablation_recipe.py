"""
Recipe Ablation: KD + Mixup — Revision Item 9
===============================================
Isolates the contribution of KD and Mixup to final accuracy.

Four conditions on ResNet-56 / CIFAR-10 / f=0.5:
  full     : KD=True,  Mixup=True   [our method]
  no_kd    : KD=False, Mixup=True
  no_mixup : KD=True,  Mixup=False
  neither  : KD=False, Mixup=False  [bare fine-tune]

Answers reviewer: is the gain from the TG criterion or the training recipe?

Usage:
    cd /arf/scratch/mhassan/transgap_v2
    python -m experiments.run_ablation_recipe \
        --condition full \
        --checkpoint_dir checkpoints \
        --data_path data \
        --save_dir checkpoints

Aggregate:
    python -m experiments.run_ablation_recipe \
        --condition aggregate --save_dir checkpoints
"""

import argparse
import os
import sys
import json
import copy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch


CONDITIONS = {
    'full':      {'use_kd': True,  'use_mixup': True},
    'no_kd':     {'use_kd': False, 'use_mixup': True},
    'no_mixup':  {'use_kd': True,  'use_mixup': False},
    'neither':   {'use_kd': False, 'use_mixup': False},
}


def run_condition(condition_name, condition, checkpoint_dir, data_path, save_dir,
                  ft_epochs=300, ft_lr=0.01, target_flops=0.5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    out_path = os.path.join(save_dir, f"recipe_{condition_name}_summary.json")
    if os.path.exists(out_path):
        print(f"SKIP (done): {out_path}")
        with open(out_path) as f:
            return json.load(f)

    from models import resnet56
    from metrics.transformation_gap import TGAnalyzer
    from metrics.safety_bounds import compute_safety_bounds
    from metrics.flops import count_flops
    from pruning.block_pruner import BlockPruner
    from pruning.channel_pruner import ChannelPruner
    from pruning.optimizer import PruningOptimizer
    from training.baseline import get_cifar_loaders
    from training.trainer import Trainer
    from utils.checkpoint import load_checkpoint

    model = resnet56(num_classes=10)
    ckpt_path = os.path.join(checkpoint_dir, 'resnet56_cifar10_baseline.pth')
    ckpt = load_checkpoint(ckpt_path, model, device=device)
    baseline_acc = ckpt.get('best_acc', 0)
    teacher = copy.deepcopy(model).eval().to(device)
    model.to(device)

    train_loader, test_loader, _ = get_cifar_loaders('cifar10', 128, data_path)

    # Pruning — identical across all conditions
    analyzer = TGAnalyzer(model, device)
    block_tg = analyzer.analyze_blocks(test_loader, num_batches=10)
    prunable = [n for n, tg in block_tg.items() if tg == tg]
    channel_tg = analyzer.analyze_channels(
        test_loader, block_names=prunable, num_batches=10, method='approx'
    )
    safety_bounds = compute_safety_bounds(
        model, block_tg, test_loader, device, num_batches=10
    )

    original_flops = count_flops(model, (1, 3, 32, 32))
    opt = PruningOptimizer(
        model, block_tg, channel_tg, safety_bounds, input_size=(1, 3, 32, 32)
    )
    blocks_to_remove, channel_ratios = opt.optimize(
        target_flops_ratio=target_flops, max_block_tg=0.05, max_channel_ratio=0.7
    )

    if blocks_to_remove:
        bp = BlockPruner(model, block_tg, safety_bounds)
        model, _ = bp.prune(num_blocks=len(blocks_to_remove))
    if channel_ratios:
        cp = ChannelPruner(model, channel_tg)
        cp.prune_all_blocks(channel_ratios)

    pruned_flops = count_flops(model, (1, 3, 32, 32))
    model.to(device)

    tg_lr_scale = {}
    if blocks_to_remove:
        for name in model.block_names:
            layer_name, idx_str = name.rsplit('.', 1)
            idx = int(idx_str)
            neighbor_tg_sum = 0
            for offset in [-2, -1, 1, 2]:
                neighbor_name = f"{layer_name}.{idx + offset}"
                if neighbor_name in blocks_to_remove:
                    neighbor_tg_sum += block_tg.get(neighbor_name, 0)
            if neighbor_tg_sum > 0:
                tg_lr_scale[name] = 1.0 + 1.0 * neighbor_tg_sum

    # Fine-tuning — only KD/Mixup changes
    ft_config = {
        'lr': ft_lr, 'momentum': 0.9, 'weight_decay': 5e-4,
        'epochs': ft_epochs, 'warmup_epochs': 5,
        'label_smoothing': 0.1 if condition['use_mixup'] else 0.0,
        'use_mixup': condition['use_mixup'],
        'mixup_alpha': 0.2,
        'use_kd': condition['use_kd'],
        'kd_alpha': 0.7,
        'kd_temperature': 4.0,
        'grad_clip': 5.0,
        'tg_lr_scale': tg_lr_scale if tg_lr_scale else None,
    }

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"recipe_{condition_name}.pth")

    trainer = Trainer(model, train_loader, test_loader, device, ft_config, teacher=teacher)
    result = trainer.train(save_path=save_path)

    summary = {
        'condition': condition_name,
        'use_kd': condition['use_kd'],
        'use_mixup': condition['use_mixup'],
        'baseline_acc': baseline_acc,
        'final_acc': result['best_acc'],
        'acc_delta': round(result['best_acc'] - baseline_acc, 2),
        'flops_reduction_pct': round((1 - pruned_flops / original_flops) * 100, 1),
        'blocks_removed': blocks_to_remove,
    }
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nDONE {condition_name}: "
          f"{result['best_acc']:.2f}% (Δ{result['best_acc']-baseline_acc:+.2f}%)")
    return summary


def aggregate(save_dir):
    print(f"\n{'='*55}")
    print("Item 9 — Recipe Ablation Summary")
    print(f"{'='*55}")
    print(f"{'Condition':<12} {'KD':>5} {'Mixup':>6} {'Acc':>8} {'Delta':>8}")
    print("-" * 44)
    for cond_name, cond in CONDITIONS.items():
        p = os.path.join(save_dir, f"recipe_{cond_name}_summary.json")
        if os.path.exists(p):
            with open(p) as f:
                r = json.load(f)
            print(f"{cond_name:<12} {str(cond['use_kd']):>5} {str(cond['use_mixup']):>6} "
                  f"{r['final_acc']:>8.2f}% {r['acc_delta']:>+8.2f}%")
        else:
            print(f"{cond_name:<12} {'':>5} {'':>6} {'MISSING':>8}")


def main():
    parser = argparse.ArgumentParser(
        description='TransGap v2 — Recipe ablation (Item 9)'
    )
    parser.add_argument('--condition',
                        choices=list(CONDITIONS.keys()) + ['all', 'aggregate'],
                        required=True)
    parser.add_argument('--ft_epochs', type=int, default=300)
    parser.add_argument('--ft_lr', type=float, default=0.01)
    parser.add_argument('--target_flops', type=float, default=0.5)
    parser.add_argument('--checkpoint_dir', default='checkpoints')
    parser.add_argument('--data_path', default='data')
    parser.add_argument('--save_dir', default='checkpoints')
    args = parser.parse_args()

    if args.condition == 'aggregate':
        aggregate(args.save_dir)
        return

    conditions = CONDITIONS if args.condition == 'all' \
                 else {args.condition: CONDITIONS[args.condition]}

    for cond_name, cond in conditions.items():
        print(f"\n{'='*50}")
        print(f"Condition: {cond_name}  (KD={cond['use_kd']}, Mixup={cond['use_mixup']})")
        run_condition(
            cond_name, cond,
            args.checkpoint_dir, args.data_path, args.save_dir,
            ft_epochs=args.ft_epochs, ft_lr=args.ft_lr,
            target_flops=args.target_flops
        )


if __name__ == '__main__':
    main()
