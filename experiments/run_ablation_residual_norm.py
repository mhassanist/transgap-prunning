"""
Ablation: TG vs ||F(X)||/||X|| — Revision Item 8
==================================================
The most important missing ablation per reviewers.

Three block-importance criteria compared under identical pipeline:
  tg     : TG = 1 - cos(X, Y)          [TransGap]
  rn     : RN = ||F(X)||/||X||          [residual norm ratio alone]
  hybrid : TG × RN                      [product]

If TG wins → directional (cosine) information matters beyond magnitude.
If RN ties → honest note, but multi-granularity story still holds.

Architecture: ResNet-56 / CIFAR-10 / f=0.5
Seeds: 42, 123, 456

Usage:
    cd /arf/scratch/mhassan/transgap_v2
    python -m experiments.run_ablation_residual_norm \
        --seed 42 \
        --checkpoint_dir checkpoints \
        --data_path data \
        --save_dir checkpoints

Aggregate:
    python -m experiments.run_ablation_residual_norm \
        --aggregate --save_dir checkpoints
"""

import argparse
import os
import sys
import json
import copy
import random
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


@torch.no_grad()
def compute_residual_norm_ratio(model, dataloader, device, num_batches=10):
    """||F(X)||/||X|| per block. Lower = lazier (same semantics as TG)."""
    model.eval()
    model.to(device)

    inputs_store = {n: [] for n in model.block_names}
    outputs_store = {n: [] for n in model.block_names}
    handles = []

    for name, block in zip(model.block_names, model.block_list):
        def make_hook(n):
            def hook(module, inp, out):
                inputs_store[n].append(inp[0].detach().cpu())
                outputs_store[n].append(out.detach().cpu())
            return hook
        handles.append(block.register_forward_hook(make_hook(name)))

    batch_count = 0
    for images, _ in dataloader:
        if batch_count >= num_batches:
            break
        model(images.to(device))
        batch_count += 1

    for h in handles:
        h.remove()

    rn = {}
    for name in model.block_names:
        if not inputs_store[name]:
            rn[name] = float('nan')
            continue
        X = torch.cat(inputs_store[name], dim=0)
        Y = torch.cat(outputs_store[name], dim=0)
        if X.shape != Y.shape:
            rn[name] = float('nan')
            continue
        F_vec = (Y - X).view(X.size(0), -1)
        X_vec = X.view(X.size(0), -1)
        ratio = (F_vec.norm(dim=1) / (X_vec.norm(dim=1) + 1e-8)).mean().item()
        rn[name] = ratio

    return rn


def run_one_criterion(criterion_name, block_scores, model, teacher,
                      train_loader, test_loader, device,
                      target_flops, ft_epochs, ft_lr, save_dir, seed):
    """Prune + fine-tune with a given block ranking."""
    from metrics.transformation_gap import TGAnalyzer
    from metrics.safety_bounds import compute_safety_bounds
    from metrics.flops import count_flops
    from pruning.block_pruner import BlockPruner
    from pruning.channel_pruner import ChannelPruner
    from pruning.optimizer import PruningOptimizer
    from training.trainer import Trainer

    work_model = copy.deepcopy(model)
    work_model.to(device)

    # Compute channel TG and safety bounds on the working model
    analyzer = TGAnalyzer(work_model, device)
    channel_tg = analyzer.analyze_channels(
        test_loader, num_batches=10, method='approx'
    )
    safety_bounds = compute_safety_bounds(
        work_model, block_scores, test_loader, device, num_batches=10
    )

    original_flops = count_flops(work_model, (1, 3, 32, 32))
    opt = PruningOptimizer(
        work_model, block_scores, channel_tg, safety_bounds,
        input_size=(1, 3, 32, 32)
    )

    # Use a generous max_block_tg — let the optimizer pick based on scores
    max_score = max(v for v in block_scores.values() if v == v)
    blocks_to_remove, channel_ratios = opt.optimize(
        target_flops_ratio=target_flops,
        max_block_tg=max_score,
        max_channel_ratio=0.7
    )

    if blocks_to_remove:
        bp = BlockPruner(work_model, block_scores, safety_bounds)
        work_model, _ = bp.prune(num_blocks=len(blocks_to_remove))
    if channel_ratios:
        cp = ChannelPruner(work_model, channel_tg)
        cp.prune_all_blocks(channel_ratios)

    pruned_flops = count_flops(work_model, (1, 3, 32, 32))
    work_model.to(device)

    tg_lr_scale = {}
    if blocks_to_remove:
        for name in work_model.block_names:
            layer_name, idx_str = name.rsplit('.', 1)
            idx = int(idx_str)
            neighbor_tg_sum = 0
            for offset in [-2, -1, 1, 2]:
                neighbor_name = f"{layer_name}.{idx + offset}"
                if neighbor_name in blocks_to_remove:
                    neighbor_tg_sum += block_scores.get(neighbor_name, 0)
            if neighbor_tg_sum > 0:
                tg_lr_scale[name] = 1.0 + 1.0 * neighbor_tg_sum

    ft_config = {
        'lr': ft_lr, 'momentum': 0.9, 'weight_decay': 5e-4,
        'epochs': ft_epochs, 'warmup_epochs': 5,
        'label_smoothing': 0.1, 'use_mixup': True, 'mixup_alpha': 0.2,
        'use_kd': True, 'kd_alpha': 0.7, 'kd_temperature': 4.0, 'grad_clip': 5.0,
        'tg_lr_scale': tg_lr_scale if tg_lr_scale else None,
    }

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"resnorm_{criterion_name}_seed{seed}.pth")
    trainer = Trainer(work_model, train_loader, test_loader, device, ft_config, teacher=teacher)
    result = trainer.train(save_path=save_path)

    return {
        'criterion': criterion_name, 'seed': seed,
        'final_acc': result['best_acc'],
        'flops_reduction_pct': round((1 - pruned_flops / original_flops) * 100, 1),
        'blocks_removed': blocks_to_remove,
    }


def main():
    parser = argparse.ArgumentParser(
        description='TransGap v2 — TG vs Residual Norm (Item 8)'
    )
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--target_flops', type=float, default=0.5)
    parser.add_argument('--ft_epochs', type=int, default=300)
    parser.add_argument('--ft_lr', type=float, default=0.01)
    parser.add_argument('--checkpoint_dir', default='checkpoints')
    parser.add_argument('--data_path', default='data')
    parser.add_argument('--save_dir', default='checkpoints')
    parser.add_argument('--aggregate', action='store_true')
    args = parser.parse_args()

    if args.aggregate:
        seeds = [42, 123, 456]
        criteria = ['tg', 'rn', 'hybrid']
        agg = {}
        for crit in criteria:
            accs = []
            for seed in seeds:
                p = os.path.join(args.save_dir, f"resnorm_{crit}_seed{seed}_result.json")
                if os.path.exists(p):
                    with open(p) as f:
                        accs.append(json.load(f)['final_acc'])
            if accs:
                agg[crit] = {
                    'mean_acc': float(np.mean(accs)),
                    'std_acc': float(np.std(accs)),
                    'n': len(accs),
                }
        print("\n=== Item 8 — TG vs Residual Norm ===")
        print(f"{'Criterion':<10} {'Mean Acc':>10} {'Std':>8} {'N':>4}")
        print("-" * 36)
        for crit, r in agg.items():
            print(f"{crit:<10} {r['mean_acc']:>10.2f}% {r['std_acc']:>7.2f}% {r['n']:>4}")
        out = os.path.join(args.save_dir, 'resnorm_aggregate.json')
        with open(out, 'w') as f:
            json.dump(agg, f, indent=2)
        print(f"\nSaved: {out}")
        return

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    from models import resnet56
    from metrics.transformation_gap import TGAnalyzer
    from training.baseline import get_cifar_loaders
    from utils.checkpoint import load_checkpoint

    model = resnet56(num_classes=10)
    ckpt_path = os.path.join(args.checkpoint_dir, 'resnet56_cifar10_baseline.pth')
    ckpt = load_checkpoint(ckpt_path, model, device=device)
    baseline_acc = ckpt.get('best_acc', 0)
    print(f"Baseline: {baseline_acc:.2f}%  |  seed={args.seed}")

    teacher = copy.deepcopy(model).eval().to(device)
    model.to(device)

    train_loader, test_loader, _ = get_cifar_loaders('cifar10', 128, args.data_path)

    # TG scores
    analyzer = TGAnalyzer(model, device)
    block_tg = analyzer.analyze_blocks(test_loader, num_batches=10)

    # Residual norm scores
    print("Computing residual norm ratios...")
    rn_scores = compute_residual_norm_ratio(model, test_loader, device, num_batches=10)

    # Hybrid = TG * RN
    hybrid = {}
    for name in block_tg:
        tg = block_tg[name]
        rn = rn_scores.get(name, float('nan'))
        hybrid[name] = tg * rn if (tg == tg and rn == rn) else float('nan')

    print(f"\n{'Block':<15} {'TG':>12} {'RN':>12} {'Hybrid':>14}")
    print("-" * 56)
    for name in sorted(block_tg.keys()):
        tg = block_tg[name]
        rn = rn_scores.get(name, float('nan'))
        hy = hybrid.get(name, float('nan'))
        print(f"{name:<15} "
              f"{'NaN':>12}" if tg != tg else f"{tg:>12.6f} "
              f"{'NaN':>12}" if rn != rn else f"{rn:>12.6f} "
              f"{'NaN':>14}" if hy != hy else f"{hy:>14.8f}")

    all_results = []
    for crit_name, scores in [('tg', block_tg), ('rn', rn_scores), ('hybrid', hybrid)]:
        out_path = os.path.join(args.save_dir, f"resnorm_{crit_name}_seed{args.seed}_result.json")
        if os.path.exists(out_path):
            print(f"SKIP {crit_name} (done): {out_path}")
            with open(out_path) as f:
                result = json.load(f)
        else:
            print(f"\n{'='*50}")
            print(f"Running criterion: {crit_name}")
            result = run_one_criterion(
                crit_name, scores, model, teacher,
                train_loader, test_loader, device,
                args.target_flops, args.ft_epochs, args.ft_lr,
                args.save_dir, args.seed
            )
            result['baseline_acc'] = baseline_acc
            result['acc_delta'] = round(result['final_acc'] - baseline_acc, 2)
            with open(out_path, 'w') as f:
                json.dump(result, f, indent=2)

        all_results.append(result)

    print(f"\n{'='*60}")
    print(f"ITEM 8 SUMMARY — seed={args.seed}")
    print(f"{'='*60}")
    print(f"{'Crit':<10} {'Acc':>8} {'Delta':>8} {'FLOPs↓':>8}")
    print("-" * 38)
    for r in all_results:
        print(f"{r['criterion']:<10} {r['final_acc']:>8.2f}% "
              f"{r.get('acc_delta', 0):>+8.2f}% {r['flops_reduction_pct']:>7.1f}%")


if __name__ == '__main__':
    main()
