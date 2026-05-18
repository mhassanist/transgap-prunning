"""
Multi-Seed Pruning Runs — Revision Item 6
==========================================
Runs the full pruning pipeline 3x with different seeds on 5 primary
operating points to add mean ± std to Tables 2–5.

Seeds affect fine-tuning randomness only (data augmentation, weight init).
TG scores and pruning decisions are deterministic.

Settings:
  r56_cifar10_f05  — ResNet-56  / CIFAR-10  / f=0.5  ★
  r56_cifar10_f06  — ResNet-56  / CIFAR-10  / f=0.6
  r110_cifar10_f06 — ResNet-110 / CIFAR-10  / f=0.6  ★
  r50_imagenet_f06 — ResNet-50  / ImageNet  / f=0.6  ★
  r50_imagenet_f05 — ResNet-50  / ImageNet  / f=0.5

Seeds: 42, 123, 456

CIFAR usage:
    cd /arf/scratch/mhassan/transgap_v2
    python -m experiments.run_multiseed \
        --setting r56_cifar10_f05 --seed 42 \
        --data_path data \
        --save_dir checkpoints

ImageNet usage:
    python -m experiments.run_multiseed \
        --setting r50_imagenet_f06 --seed 42 \
        --imagenet_path /arf/scratch/mhassan/imagenet \
        --save_dir checkpoints

Aggregate after all seeds:
    python -m experiments.run_multiseed \
        --setting aggregate_all --aggregate --save_dir checkpoints
"""

import argparse
from experiments.run_imagenet import IdentityBottleneck
import os
import sys
import json
import copy
import random
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.backends.cudnn as cudnn


# ── Settings ──────────────────────────────────────────────────────────────────

SETTINGS = {
    'r56_cifar10_f05': {
        'model': 'resnet56', 'dataset': 'cifar10',
        'target_flops': 0.5, 'ft_epochs': 300, 'ft_lr': 0.01,
        'checkpoint': 'resnet56_cifar10_baseline.pth',
        'imagenet': False,
    },
    'r56_cifar10_f06': {
        'model': 'resnet56', 'dataset': 'cifar10',
        'target_flops': 0.6, 'ft_epochs': 300, 'ft_lr': 0.01,
        'checkpoint': 'resnet56_cifar10_baseline.pth',
        'imagenet': False,
    },
    'r110_cifar10_f06': {
        'model': 'resnet110', 'dataset': 'cifar10',
        'target_flops': 0.6, 'ft_epochs': 300, 'ft_lr': 0.01,
        'checkpoint': 'resnet110_cifar10_baseline.pth',
        'imagenet': False,
    },
    'r50_imagenet_f06': {
        'target_flops': 0.6, 'ft_epochs': 90, 'ft_lr': 0.001,
        'pruned_state': 'resnet50_imagenet_f0.6_pruned_state.pth',
        'orig_summary': 'resnet50_imagenet_f0.6_summary.json',
        'imagenet': True,
    },
    'r50_imagenet_f05': {
        'target_flops': 0.5, 'ft_epochs': 90, 'ft_lr': 0.001,
        'pruned_state': 'resnet50_imagenet_f0.5_pruned_state.pth',
        'orig_summary': 'resnet50_imagenet_f0.5_summary.json',
        'imagenet': True,
    },
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    print(f"Seed: {seed}")


# ── CIFAR ─────────────────────────────────────────────────────────────────────

def run_cifar(setting, setting_name, seed, checkpoint_dir, data_path, save_dir):
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    out_path = os.path.join(save_dir, f"multiseed_{setting_name}_seed{seed}_summary.json")
    if os.path.exists(out_path):
        print(f"SKIP (done): {out_path}")
        with open(out_path) as f:
            return json.load(f)

    from models import resnet56, resnet110
    from metrics.transformation_gap import TGAnalyzer
    from metrics.safety_bounds import compute_safety_bounds
    from metrics.flops import count_flops
    from pruning.block_pruner import BlockPruner
    from pruning.channel_pruner import ChannelPruner
    from pruning.optimizer import PruningOptimizer
    from training.baseline import get_cifar_loaders
    from training.trainer import Trainer
    from utils.checkpoint import load_checkpoint

    num_classes = 10 if setting['dataset'] == 'cifar10' else 100
    model = resnet56(num_classes=num_classes) if setting['model'] == 'resnet56' \
            else resnet110(num_classes=num_classes)

    ckpt_path = os.path.join(checkpoint_dir, setting['checkpoint'])
    ckpt = load_checkpoint(ckpt_path, model, device=device)
    baseline_acc = ckpt.get('best_acc', 0)
    print(f"Baseline: {baseline_acc:.2f}%")

    teacher = copy.deepcopy(model).eval().to(device)
    model.to(device)

    train_loader, test_loader, _ = get_cifar_loaders(setting['dataset'], 128, data_path)

    # TG is deterministic — independent of seed
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
        target_flops_ratio=setting['target_flops'],
        max_block_tg=0.05, max_channel_ratio=0.7
    )

    if blocks_to_remove:
        bp = BlockPruner(model, block_tg, safety_bounds)
        model, _ = bp.prune(num_blocks=len(blocks_to_remove))
    if channel_ratios:
        cp = ChannelPruner(model, channel_tg)
        cp.prune_all_blocks(channel_ratios)

    pruned_flops = count_flops(model, (1, 3, 32, 32))
    model.to(device)

    # TG-aware LR scaling for blocks adjacent to removed ones (matches original recipe)
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

    ft_config = {
        'lr': setting['ft_lr'], 'momentum': 0.9, 'weight_decay': 5e-4,
        'epochs': setting['ft_epochs'], 'warmup_epochs': 5,
        'label_smoothing': 0.1, 'use_mixup': True, 'mixup_alpha': 0.2,
        'use_kd': True, 'kd_alpha': 0.7, 'kd_temperature': 4.0, 'grad_clip': 5.0,
        'tg_lr_scale': tg_lr_scale if tg_lr_scale else None,
    }

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"multiseed_{setting_name}_seed{seed}.pth")
    trainer = Trainer(model, train_loader, test_loader, device, ft_config, teacher=teacher)
    result = trainer.train(save_path=save_path)

    summary = {
        'setting': setting_name, 'seed': seed,
        'baseline_acc': baseline_acc,
        'final_acc': result['best_acc'],
        'acc_delta': round(result['best_acc'] - baseline_acc, 2),
        'flops_reduction_pct': round((1 - pruned_flops / original_flops) * 100, 1),
        'blocks_removed': blocks_to_remove,
    }
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*50}")
    print(f"DONE: {setting_name} seed={seed}")
    print(f"  Acc   : {result['best_acc']:.2f}%  (Δ{result['best_acc']-baseline_acc:+.2f}%)")
    print(f"  FLOPs↓: {summary['flops_reduction_pct']:.1f}%")
    print(f"  Saved : {out_path}")
    return summary


# ── ImageNet ──────────────────────────────────────────────────────────────────

def run_imagenet(setting, setting_name, seed, checkpoint_dir, imagenet_path, save_dir):
    """
    Re-fine-tune from the existing pruned state with a different seed.
    Block removal and channel ratios are already fixed in pruned_state.pth.
    Only fine-tuning randomness varies across seeds.
    """
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    out_path = os.path.join(save_dir, f"multiseed_{setting_name}_seed{seed}_summary.json")
    if os.path.exists(out_path):
        print(f"SKIP (done): {out_path}")
        with open(out_path) as f:
            return json.load(f)

    # Original run metadata
    orig_summary_path = os.path.join(checkpoint_dir, setting['orig_summary'])
    with open(orig_summary_path) as f:
        orig = json.load(f)
    baseline_acc = orig['baseline_acc']
    flops_reduction = orig['flops_reduction_pct']

    # Load pruned model structure (saved as full model object)
    pruned_state_path = os.path.join(checkpoint_dir, setting['pruned_state'])
    state = torch.load(pruned_state_path, map_location=device, weights_only=False)
    model = state['model']
    model.to(device)

    # Teacher: fresh torchvision weights
    from torchvision.models import resnet50, ResNet50_Weights
    teacher = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).to(device).eval()

    # Data
    from experiments.run_imagenet import get_imagenet_loaders
    train_loader, val_loader = get_imagenet_loaders(imagenet_path, batch_size=64, num_workers=8)

    ft_config = {
        'lr': setting['ft_lr'], 'momentum': 0.9, 'weight_decay': 1e-4,
        'epochs': setting['ft_epochs'], 'warmup_epochs': 5,
        'label_smoothing': 0.1, 'use_mixup': False,
        'use_kd': True, 'kd_alpha': 0.5, 'kd_temperature': 4.0, 'grad_clip': 0,
    }

    from training.trainer import Trainer
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"multiseed_{setting_name}_seed{seed}.pth")
    resume_path = save_path.replace('.pth', '_resume.pth')
    trainer = Trainer(model, train_loader, val_loader, device, ft_config, teacher=teacher)
    if os.path.exists(resume_path):
        print(f"Resuming from: {resume_path}")
        trainer.resume_from(resume_path)
    result = trainer.train(save_path=save_path)

    summary = {
        'setting': setting_name, 'seed': seed,
        'baseline_acc': baseline_acc,
        'final_acc': result['best_acc'],
        'acc_delta': round(result['best_acc'] - baseline_acc, 2),
        'flops_reduction_pct': flops_reduction,
    }
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*50}")
    print(f"DONE: {setting_name} seed={seed}")
    print(f"  Acc   : {result['best_acc']:.2f}%  (Δ{result['best_acc']-baseline_acc:+.2f}%)")
    print(f"  Saved : {out_path}")
    return summary


# ── Aggregation ───────────────────────────────────────────────────────────────

def aggregate(setting_name, save_dir):
    seeds = [42, 123, 456]
    accs, deltas, flops_list = [], [], []
    for seed in seeds:
        p = os.path.join(save_dir, f"multiseed_{setting_name}_seed{seed}_summary.json")
        if os.path.exists(p):
            with open(p) as f:
                s = json.load(f)
            accs.append(s['final_acc'])
            deltas.append(s['acc_delta'])
            flops_list.append(s['flops_reduction_pct'])

    if not accs:
        print(f"  {setting_name}: no results found")
        return None

    accs = np.array(accs)
    deltas = np.array(deltas)
    agg = {
        'setting': setting_name,
        'n_seeds': len(accs),
        'mean_acc': float(np.mean(accs)),
        'std_acc': float(np.std(accs)),
        'mean_delta': float(np.mean(deltas)),
        'std_delta': float(np.std(deltas)),
        'flops_reduction_pct': float(np.mean(flops_list)),
        'per_seed': {str(s): float(a) for s, a in zip(seeds[:len(accs)], accs)},
    }
    out = os.path.join(save_dir, f"multiseed_{setting_name}_aggregate.json")
    with open(out, 'w') as f:
        json.dump(agg, f, indent=2)
    print(f"  {setting_name}: {agg['mean_acc']:.2f}% ± {agg['std_acc']:.2f}%  "
          f"(Δ{agg['mean_delta']:+.2f}% ± {agg['std_delta']:.2f}%)  "
          f"FLOPs↓{agg['flops_reduction_pct']:.1f}%")
    return agg


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='TransGap v2 — Multi-seed runs (Item 6)')
    parser.add_argument('--setting', required=True,
                        choices=list(SETTINGS.keys()) + ['aggregate_all'])
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed: 42, 123, or 456')
    parser.add_argument('--checkpoint_dir', default='checkpoints')
    parser.add_argument('--data_path', default='data')
    parser.add_argument('--imagenet_path', default='/arf/scratch/mhassan/imagenet')
    parser.add_argument('--save_dir', default='checkpoints')
    parser.add_argument('--aggregate', action='store_true',
                        help='Aggregate existing seed results')
    args = parser.parse_args()

    if args.aggregate or args.setting == 'aggregate_all':
        names = list(SETTINGS.keys()) if args.setting == 'aggregate_all' \
                else [args.setting]
        print("\n=== Multi-seed Aggregate Results ===")
        for n in names:
            aggregate(n, args.save_dir)
        return

    s = SETTINGS[args.setting]
    if s['imagenet']:
        run_imagenet(s, args.setting, args.seed,
                     args.checkpoint_dir, args.imagenet_path, args.save_dir)
    else:
        run_cifar(s, args.setting, args.seed,
                  args.checkpoint_dir, args.data_path, args.save_dir)


if __name__ == '__main__':
    main()
