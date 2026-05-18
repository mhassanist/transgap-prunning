"""
Eq.9 Approximation vs Exact + Calibration Subset Sensitivity
=============================================================
Item 10: Scatter of approx vs exact channel TG (validates Eq.9).
Item 11: 5 random 128-image subsets → std across subsets (validates A7).

Both are fast — no fine-tuning needed. Runs in under 2 hours.

Usage:
    cd /arf/scratch/mhassan/transgap_v2
    python -m experiments.run_ablation_channel_exact \
        --experiment both \
        --checkpoint_dir checkpoints \
        --data_path data \
        --save_dir checkpoints
"""

import argparse
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch


# ── Item 10: Approx vs Exact channel TG ──────────────────────────────────────

def run_item10(model, test_loader, device, save_dir):
    from metrics.transformation_gap import compute_channel_tg

    blocks = ['layer1.2', 'layer2.1', 'layer3.2']
    print(f"\n{'='*60}")
    print("Item 10: Eq.9 Approx vs Exact channel TG")
    print(f"{'='*60}")

    results = {}
    for block_name in blocks:
        print(f"\n  Block: {block_name}")

        tg_approx = compute_channel_tg(
            model, test_loader, device, block_name, num_batches=10, method='approx'
        )
        print(f"  Approx done ({len(tg_approx)} channels)")

        tg_exact = compute_channel_tg(
            model, test_loader, device, block_name, num_batches=10, method='exact'
        )
        print(f"  Exact done")

        # Normalise to [0,1] for fair comparison
        def norm01(t):
            mn, mx = t.min(), t.max()
            return (t - mn) / (mx - mn + 1e-8)

        a = norm01(tg_approx).cpu().numpy()
        e = norm01(tg_exact).cpu().numpy()

        try:
            from scipy.stats import spearmanr, pearsonr
            sp_r, sp_p = spearmanr(a, e)
            pe_r, pe_p = pearsonr(a, e)
        except ImportError:
            def rank_corr(x, y):
                rx = x.argsort().argsort().astype(float)
                ry = y.argsort().argsort().astype(float)
                return np.corrcoef(rx, ry)[0, 1]
            sp_r = rank_corr(a, e)
            pe_r = np.corrcoef(a, e)[0, 1]
            sp_p = pe_p = float('nan')

        k = max(1, len(a) // 4)
        top_approx = set(np.argsort(a)[-k:])
        top_exact = set(np.argsort(e)[-k:])
        topk_agree = len(top_approx & top_exact) / k

        print(f"  Spearman r={sp_r:.4f}  Pearson r={pe_r:.4f}  Top-{k} agree={topk_agree:.2%}")

        results[block_name] = {
            'n_channels': int(len(a)),
            'spearman_r': float(sp_r),
            'pearson_r': float(pe_r),
            'topk_agreement': float(topk_agree),
            'k': int(k),
            'approx_values': a.tolist(),
            'exact_values': e.tolist(),
        }

    os.makedirs(save_dir, exist_ok=True)
    out = os.path.join(save_dir, 'ablation_channel_tg_approx_vs_exact.json')
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out}")

    print(f"\n{'='*60}")
    print("Item 10 Summary")
    print(f"{'='*60}")
    print(f"{'Block':<15} {'Channels':>8} {'Spearman':>10} {'Pearson':>8} {'Top-k%':>8}")
    print("-" * 54)
    for name, r in results.items():
        print(f"{name:<15} {r['n_channels']:>8} {r['spearman_r']:>10.4f} "
              f"{r['pearson_r']:>8.4f} {r['topk_agreement']:>7.2%}")
    return results


# ── Item 11: Calibration subset sensitivity ───────────────────────────────────

def run_item11(model, test_loader, device, save_dir, n_subsets=5):
    from metrics.transformation_gap import compute_block_tg

    print(f"\n{'='*60}")
    print(f"Item 11: Calibration Subset Sensitivity ({n_subsets} subsets × 128 images)")
    print(f"{'='*60}")

    # Collect test images for random subsampling
    all_images, all_labels = [], []
    for imgs, lbls in test_loader:
        all_images.append(imgs)
        all_labels.append(lbls)
        if sum(x.size(0) for x in all_images) >= n_subsets * 128 * 3:
            break
    all_images = torch.cat(all_images, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    total = all_images.size(0)
    print(f"  Available test images: {total}")

    subset_results = []
    for i in range(n_subsets):
        seed = 2000 + i
        torch.manual_seed(seed)
        np.random.seed(seed)

        idx = torch.randperm(total)[:128]
        sub_imgs = all_images[idx].to(device)
        sub_lbls = all_labels[idx]

        sub_ds = torch.utils.data.TensorDataset(sub_imgs.cpu(), sub_lbls)
        sub_loader = torch.utils.data.DataLoader(sub_ds, batch_size=128, shuffle=False)

        tg = compute_block_tg(model, sub_loader, device, num_batches=1)
        prunable = {k: v for k, v in tg.items() if v == v}
        subset_results.append(prunable)

        ranked = sorted(prunable.items(), key=lambda x: x[1])[:5]
        print(f"  Subset {i+1} (seed={seed}): "
              f"top-5 laziest = {[(n, f'{t:.6f}') for n, t in ranked]}")

    # Stats across subsets
    block_names = sorted(subset_results[0].keys())
    per_block = {n: [sr[n] for sr in subset_results if n in sr] for n in block_names}

    stats = {}
    for name, vals in per_block.items():
        v = np.array(vals)
        stats[name] = {
            'mean': float(np.mean(v)), 'std': float(np.std(v)),
            'cv': float(np.std(v) / (np.mean(v) + 1e-8)),
            'values': v.tolist(),
        }

    # Rank stability
    orderings = [
        [n for n, _ in sorted([(n, sr[n]) for n in block_names], key=lambda x: x[1])]
        for sr in subset_results
    ]
    ref = orderings[0]
    top5_stable = all(o[:5] == ref[:5] for o in orderings)
    top10_stable = all(o[:10] == ref[:10] for o in orderings)

    max_cv = max(s['cv'] for s in stats.values())
    mean_std = float(np.mean([s['std'] for s in stats.values()]))

    print(f"\n  Top-5 rank stable: {top5_stable}")
    print(f"  Top-10 rank stable: {top10_stable}")
    print(f"  Max CV: {max_cv:.4f}   Mean std: {mean_std:.6f}")

    print(f"\n  {'Block':<15} {'Mean':>10} {'Std':>10} {'CV':>8}")
    print("  " + "-" * 46)
    for name in sorted(stats.keys()):
        s = stats[name]
        print(f"  {name:<15} {s['mean']:>10.6f} {s['std']:>10.6f} {s['cv']:>8.4f}")

    result = {
        'n_subsets': n_subsets, 'subset_size': 128,
        'block_stats': stats,
        'top5_rank_stable': top5_stable,
        'top10_rank_stable': top10_stable,
        'max_cv': float(max_cv),
        'mean_std': mean_std,
    }
    os.makedirs(save_dir, exist_ok=True)
    out = os.path.join(save_dir, 'ablation_calibration_subset_sensitivity.json')
    with open(out, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {out}")
    return result


def main():
    parser = argparse.ArgumentParser(
        description='TransGap v2 — Channel approx + calibration sensitivity (Items 10+11)'
    )
    parser.add_argument('--experiment',
                        choices=['item10', 'item11', 'both'], default='both')
    parser.add_argument('--checkpoint_dir', default='checkpoints')
    parser.add_argument('--data_path', default='data')
    parser.add_argument('--save_dir', default='checkpoints')
    parser.add_argument('--n_subsets', type=int, default=5)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    from models import resnet56
    from training.baseline import get_cifar_loaders
    from utils.checkpoint import load_checkpoint

    model = resnet56(num_classes=10)
    ckpt_path = os.path.join(args.checkpoint_dir, 'resnet56_cifar10_baseline.pth')
    load_checkpoint(ckpt_path, model, device=device)
    model.to(device).eval()

    _, test_loader, _ = get_cifar_loaders('cifar10', 128, args.data_path)

    if args.experiment in ('item10', 'both'):
        run_item10(model, test_loader, device, args.save_dir)

    if args.experiment in ('item11', 'both'):
        run_item11(model, test_loader, device, args.save_dir, n_subsets=args.n_subsets)


if __name__ == '__main__':
    main()
