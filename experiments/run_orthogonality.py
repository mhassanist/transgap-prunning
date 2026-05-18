"""
Orthogonality Validation — Revision Items 7 + 12
==================================================
Validates the small-residual orthogonality assumption from Eq.5:

  1 - cos(X, X+F(X)) ≈ ||F(X)||² / (2·||X||²)

This requires <X, F(X)> ≈ 0 (X and F(X) approximately orthogonal).

Computes cos(X, F(X)) per block across the calibration set for:
  - ResNet-56  / CIFAR-10  (layer1.5 is the known outlier)
  - ResNet-110 / CIFAR-10
  - ResNet-50  / ImageNet

Item 7:  Violin plots across all blocks for 3 backbones.
Item 12: Deep-dive on layer1.5 — why the bound is loose there.

Output JSON files saved to save_dir:
  orthogonality_r56_cifar10.json
  orthogonality_r110_cifar10.json
  orthogonality_r50_imagenet.json
  orthogonality_layer1p5_deepdive_r56_cifar10.json

Usage (run from code dir):
    cd /arf/scratch/mhassan/transgap_v2
    python -m experiments.run_orthogonality \
        --backbone r56_cifar10 \
        --checkpoint_dir checkpoints \
        --data_path data \
        --save_dir checkpoints
"""

import argparse
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F


@torch.no_grad()
def compute_orthogonality(model, dataloader, device, num_batches=10):
    """
    For each prunable block compute per-sample cos(X, F(X)).

    F(X) = Y - X  where Y is the block output and X is the block input.

    Returns dict: block_name -> {
        cos_XF_mean, cos_XF_abs_mean, cos_XF_std,
        residual_ratio_mean,   # ||F(X)||/||X||
        tg_mean,               # 1 - cos(X, Y)  (cross-check)
        orthogonality_pct,     # % samples with |cos(X,F(X))| < 0.1
        cos_XF_samples,        # first 200 sample values for plotting
    }
    """
    model.eval()
    model.to(device)

    block_names = model.block_names
    blocks = model.block_list

    inputs_store = {name: [] for name in block_names}
    outputs_store = {name: [] for name in block_names}
    handles = []

    for name, block in zip(block_names, blocks):
        def make_hook(n):
            def hook(module, inp, out):
                inputs_store[n].append(inp[0].detach().cpu())
                outputs_store[n].append(out.detach().cpu())
            return hook
        h = block.register_forward_hook(make_hook(name))
        handles.append(h)

    batch_count = 0
    for images, _ in dataloader:
        if batch_count >= num_batches:
            break
        images = images.to(device)
        model(images)
        batch_count += 1

    for h in handles:
        h.remove()

    results = {}
    for name in block_names:
        if not inputs_store[name]:
            continue

        all_X = torch.cat(inputs_store[name], dim=0)
        all_Y = torch.cat(outputs_store[name], dim=0)

        # Downsampling blocks have shape mismatch — skip
        if all_X.shape != all_Y.shape:
            results[name] = {'skipped': True, 'reason': 'downsampling'}
            continue

        all_F = all_Y - all_X  # residual function output

        X_flat = all_X.view(all_X.size(0), -1)
        F_flat = all_F.view(all_F.size(0), -1)
        Y_flat = all_Y.view(all_Y.size(0), -1)

        # cos(X, F(X)) — the orthogonality check
        cos_XF = F.cosine_similarity(X_flat, F_flat, dim=1)

        # ||F(X)||/||X|| — relative residual magnitude
        F_norm = F_flat.norm(dim=1)
        X_norm = X_flat.norm(dim=1)
        residual_ratio = F_norm / (X_norm + 1e-8)

        # TG = 1 - cos(X, Y) — sanity cross-check
        cos_XY = F.cosine_similarity(X_flat, Y_flat, dim=1)
        tg = 1.0 - cos_XY

        orth_pct = (cos_XF.abs() < 0.1).float().mean().item() * 100

        results[name] = {
            'cos_XF_mean': float(cos_XF.mean()),
            'cos_XF_abs_mean': float(cos_XF.abs().mean()),
            'cos_XF_std': float(cos_XF.std()),
            'residual_ratio_mean': float(residual_ratio.mean()),
            'residual_ratio_std': float(residual_ratio.std()),
            'tg_mean': float(tg.mean()),
            'tg_std': float(tg.std()),
            'orthogonality_pct': float(orth_pct),
            'n_samples': int(all_X.size(0)),
            # Store first 200 samples for violin plots
            'cos_XF_samples': cos_XF[:200].tolist(),
        }

    return results


def deepdive_layer1p5(results):
    """Extract layer1.5 diagnostics and compare to the laziest block."""
    key = None
    for k in results:
        if '1.5' in k and not results[k].get('skipped'):
            key = k
            break
    if key is None:
        return {'error': 'layer1.5 not found or skipped'}

    r = results[key]
    out = {
        'block': key,
        'cos_XF_abs_mean': r['cos_XF_abs_mean'],
        'cos_XF_std': r['cos_XF_std'],
        'orthogonality_pct': r['orthogonality_pct'],
        'residual_ratio_mean': r['residual_ratio_mean'],
        'tg_mean': r['tg_mean'],
        'interpretation': (
            f"At block {key}: mean |cos(X, F(X))| = {r['cos_XF_abs_mean']:.4f}, "
            f"orthogonality holds (<0.1) for {r['orthogonality_pct']:.1f}% of samples. "
            "When |cos(X,F(X))| is large the second-order Taylor approximation in Eq.5 "
            "underestimates ||F(X)||, causing the Lipschitz bound to be loose."
        ),
    }
    # Contrast with laziest block (layer1.1)
    lazy = None
    for k in results:
        if '1.1' in k and not results[k].get('skipped'):
            lazy = k
            break
    if lazy:
        out['comparison_layer1p1'] = {
            'block': lazy,
            'cos_XF_abs_mean': results[lazy]['cos_XF_abs_mean'],
            'orthogonality_pct': results[lazy]['orthogonality_pct'],
            'tg_mean': results[lazy]['tg_mean'],
        }
    return out


def print_summary(results, backbone):
    print(f"\n{'='*70}")
    print(f"Orthogonality — {backbone}")
    print(f"{'='*70}")
    print(f"{'Block':<15} {'TG':>8} {'|cos(X,F)| mean':>17} {'Orth%':>8}")
    print("-" * 52)
    for name in sorted(results.keys()):
        r = results[name]
        if r.get('skipped'):
            print(f"{name:<15}  SKIP (downsample)")
            continue
        print(f"{name:<15} {r['tg_mean']:>8.6f} {r['cos_XF_abs_mean']:>17.6f} "
              f"{r['orthogonality_pct']:>7.1f}%")


def run_cifar_backbone(backbone, model_cls, ckpt_name, checkpoint_dir, data_path, save_dir):
    from training.baseline import get_cifar_loaders
    from utils.checkpoint import load_checkpoint
    import importlib

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model_cls(num_classes=10)
    ckpt_path = os.path.join(checkpoint_dir, ckpt_name)
    load_checkpoint(ckpt_path, model, device=device)

    _, test_loader, _ = get_cifar_loaders('cifar10', 128, data_path)

    print(f"\nComputing orthogonality: {backbone}...")
    results = compute_orthogonality(model, test_loader, device, num_batches=10)
    print_summary(results, backbone)

    # Save
    os.makedirs(save_dir, exist_ok=True)
    save_results = {}
    for name, r in results.items():
        if r.get('skipped'):
            save_results[name] = r
        else:
            save_results[name] = {k: v for k, v in r.items() if k != 'cos_XF_samples'}
            save_results[name]['cos_XF_samples'] = r['cos_XF_samples']

    out_path = os.path.join(save_dir, f"orthogonality_{backbone}.json")
    with open(out_path, 'w') as f:
        json.dump(save_results, f, indent=2)
    print(f"Saved: {out_path}")

    # Layer1.5 deep-dive
    dd = deepdive_layer1p5(results)
    dd_path = os.path.join(save_dir, f"orthogonality_layer1p5_deepdive_{backbone}.json")
    with open(dd_path, 'w') as f:
        json.dump(dd, f, indent=2)
    print(f"Layer1.5 deep-dive: {dd_path}")
    print(f"  → {dd.get('interpretation', 'N/A')[:120]}...")
    return results


def run_imagenet_backbone(checkpoint_dir, imagenet_path, save_dir):
    """Orthogonality for ResNet-50 using the run_imagenet TG functions."""
    from experiments.run_imagenet import (
        get_imagenet_loaders, compute_resnet50_block_tg
    )
    from torchvision.models import resnet50, ResNet50_Weights

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).to(device)
    model.eval()

    _, val_loader = get_imagenet_loaders(imagenet_path, batch_size=64, num_workers=8)

    print("\nComputing orthogonality: r50_imagenet...")

    # For ImageNet we use the inline hook approach since ResNet-50
    # doesn't have block_names / block_list attributes
    results = {}
    for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
        layer = getattr(model, layer_name)
        for block_idx in range(len(layer)):
            name = f"{layer_name}.{block_idx}"
            block = layer[block_idx]

            inputs_store, outputs_store = [], []

            def make_hook(il, ol):
                def h(module, inp, out):
                    il.append(inp[0].detach().cpu())
                    ol.append(out.detach().cpu())
                return h

            handle = block.register_forward_hook(make_hook(inputs_store, outputs_store))

            batch_count = 0
            with torch.no_grad():
                for images, _ in val_loader:
                    if batch_count >= 10:
                        break
                    model(images.to(device))
                    batch_count += 1

            handle.remove()

            all_X = torch.cat(inputs_store, dim=0)
            all_Y = torch.cat(outputs_store, dim=0)

            if all_X.shape != all_Y.shape:
                results[name] = {'skipped': True, 'reason': 'downsampling'}
                continue

            all_F = all_Y - all_X
            X_flat = all_X.view(all_X.size(0), -1)
            F_flat = all_F.view(all_F.size(0), -1)
            Y_flat = all_Y.view(all_Y.size(0), -1)

            cos_XF = F.cosine_similarity(X_flat, F_flat, dim=1)
            residual_ratio = F_flat.norm(dim=1) / (X_flat.norm(dim=1) + 1e-8)
            cos_XY = F.cosine_similarity(X_flat, Y_flat, dim=1)
            tg = 1.0 - cos_XY
            orth_pct = (cos_XF.abs() < 0.1).float().mean().item() * 100

            results[name] = {
                'cos_XF_mean': float(cos_XF.mean()),
                'cos_XF_abs_mean': float(cos_XF.abs().mean()),
                'cos_XF_std': float(cos_XF.std()),
                'residual_ratio_mean': float(residual_ratio.mean()),
                'tg_mean': float(tg.mean()),
                'orthogonality_pct': float(orth_pct),
                'n_samples': int(all_X.size(0)),
                'cos_XF_samples': cos_XF[:200].tolist(),
            }

    print_summary(results, 'r50_imagenet')
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, 'orthogonality_r50_imagenet.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {out_path}")
    return results


def main():
    parser = argparse.ArgumentParser(
        description='TransGap v2 — Orthogonality validation (Items 7 + 12)'
    )
    parser.add_argument('--backbone',
                        choices=['r56_cifar10', 'r110_cifar10', 'r50_imagenet', 'all'],
                        default='all')
    parser.add_argument('--checkpoint_dir', default='checkpoints')
    parser.add_argument('--data_path', default='data')
    parser.add_argument('--imagenet_path', default='/arf/scratch/mhassan/imagenet')
    parser.add_argument('--save_dir', default='checkpoints')
    args = parser.parse_args()

    from models import resnet56, resnet110

    backbones = ['r56_cifar10', 'r110_cifar10', 'r50_imagenet'] \
                if args.backbone == 'all' else [args.backbone]

    for bb in backbones:
        if bb == 'r56_cifar10':
            run_cifar_backbone(bb, resnet56, 'resnet56_cifar10_baseline.pth',
                               args.checkpoint_dir, args.data_path, args.save_dir)
        elif bb == 'r110_cifar10':
            run_cifar_backbone(bb, resnet110, 'resnet110_cifar10_baseline.pth',
                               args.checkpoint_dir, args.data_path, args.save_dir)
        elif bb == 'r50_imagenet':
            if not os.path.isdir(args.imagenet_path):
                print(f"Skipping r50_imagenet: {args.imagenet_path} not found")
                continue
            run_imagenet_backbone(args.checkpoint_dir, args.imagenet_path, args.save_dir)


if __name__ == '__main__':
    main()
