"""
E1: Cross-Backbone Correlation — ResNet-110 / CIFAR-10
========================================================
Repeats the A8 block-removal study on ResNet-110 with the full 150-epoch recipe
to validate that TG predicts accuracy drop across backbones.

Pipeline:
  --scan      Compute TG for all R110 blocks, save list of blocks with TG < 0.025
  --block X   Remove block X, fine-tune 150 epochs, save per-block summary JSON
  --aggregate Read all summary JSONs, compute Pearson r (TG rank vs Δacc)

Baseline: 95.89%  (ResNet-110 / CIFAR-10)
TG cutoff: 0.025  (same threshold used on ResNet-56)
Fine-tune: 150 epochs, lr=0.01, cosine LR, KD(α=0.7 T=4), Mixup(α=0.2)

Usage:
    # Once per checkpoint — discover qualifying blocks:
    python -m experiments.run_e1_r110_correlation --scan \\
        --checkpoint /arf/scratch/mhassan/transgap_v2/checkpoints/resnet110_cifar10_baseline.pth \\
        --data_path  /arf/scratch/mhassan/cifar \\
        --save_dir   /arf/scratch/mhassan/transgap_v2/checkpoints

    # One job per block (parallelized):
    python -m experiments.run_e1_r110_correlation --block layer2.12 \\
        --checkpoint /arf/scratch/mhassan/transgap_v2/checkpoints/resnet110_cifar10_baseline.pth \\
        --data_path  /arf/scratch/mhassan/cifar \\
        --save_dir   /arf/scratch/mhassan/transgap_v2/checkpoints

    # After all jobs complete:
    python -m experiments.run_e1_r110_correlation --aggregate \\
        --save_dir /arf/scratch/mhassan/transgap_v2/checkpoints
"""

import argparse
import copy
import glob
import json
import math
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.resnet_cifar import resnet110
from metrics.transformation_gap import compute_block_tg
from training.baseline import get_cifar_loaders
from training.trainer import Trainer
from utils.checkpoint import load_checkpoint

# ── constants ──────────────────────────────────────────────────────────────
BACKBONE       = 'r110_cifar10'
BASELINE_ACC   = 95.89          # paper-reported R110/CIFAR-10 baseline
TG_THRESHOLD   = 0.025          # same cutoff used on R56
FT_EPOCHS      = 150
CALIB_BATCHES  = 20             # batches used for TG calibration (20×128 = 2560 images)
SEED           = 42


def set_seed(s: int):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ── helpers ─────────────────────────────────────────────────────────────────

def _load_pretrained(checkpoint_path: str, device: torch.device):
    model = resnet110(num_classes=10)
    load_checkpoint(checkpoint_path, model, device=device)
    model.to(device)
    return model


def _scan_path(save_dir: str) -> str:
    return os.path.join(save_dir, 'e1_r110_scan.json')


def _summary_path(save_dir: str, block_name: str) -> str:
    safe = block_name.replace('.', '_')
    return os.path.join(save_dir, f'e1_r110_{safe}_summary.json')


# ── mode: scan ──────────────────────────────────────────────────────────────

def run_scan(checkpoint_path: str, data_path: str, save_dir: str):
    """Compute block-level TG for all R110 blocks, select TG < TG_THRESHOLD."""
    set_seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Loading checkpoint: {checkpoint_path}")

    model = _load_pretrained(checkpoint_path, device)

    train_loader, test_loader, _ = get_cifar_loaders('cifar10', batch_size=128,
                                                      data_path=data_path, num_workers=4)

    print(f"Computing block TG on {CALIB_BATCHES} batches …")
    tg_scores = compute_block_tg(model, train_loader, device, num_batches=CALIB_BATCHES)

    # Filter: finite TG < threshold (NaN = downsampling block)
    qualifying = {
        name: tg for name, tg in tg_scores.items()
        if math.isfinite(tg) and tg < TG_THRESHOLD
    }
    ranked = sorted(qualifying.items(), key=lambda x: x[1])   # ascending TG

    print(f"\nAll block TG scores (R110 / CIFAR-10):")
    print(f"{'Block':<15} {'TG':>10}  {'Qualifies?':>12}")
    print('-' * 40)
    for name, tg in sorted(tg_scores.items()):
        tg_str = f"{tg:.6f}" if math.isfinite(tg) else "N/A (ds)"
        flag = "YES" if name in qualifying else ""
        print(f"  {name:<13} {tg_str:>10}  {flag:>12}")

    print(f"\nQualifying blocks (TG < {TG_THRESHOLD}): {len(ranked)}")
    for rank, (name, tg) in enumerate(ranked, 1):
        print(f"  rank {rank:2d}  {name:<14}  TG={tg:.6f}")

    scan_data = {
        'backbone': BACKBONE,
        'tg_threshold': TG_THRESHOLD,
        'calib_batches': CALIB_BATCHES,
        'all_blocks': {k: (v if math.isfinite(v) else None) for k, v in tg_scores.items()},
        'qualifying_blocks': [
            {'block': name, 'tg': tg, 'tg_rank': rank}
            for rank, (name, tg) in enumerate(ranked, 1)
        ],
    }
    os.makedirs(save_dir, exist_ok=True)
    out = _scan_path(save_dir)
    with open(out, 'w') as f:
        json.dump(scan_data, f, indent=2)
    print(f"\nScan saved: {out}")

    # Print ready-to-copy sbatch commands
    print(f"\n=== Submit fine-tune jobs (one per block) ===")
    for _, (name, _) in enumerate(ranked, 1):
        print(f"  sbatch scripts/truba_e1_r110.slurm {name}")


# ── mode: single-block fine-tune ────────────────────────────────────────────

def run_block(block_name: str, checkpoint_path: str, data_path: str,
              save_dir: str, seed: int = SEED):
    """Remove one block, fine-tune 150 epochs, record Δacc."""
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Block: {block_name} | Backbone: {BACKBONE}")

    # ── load pretrained ──────────────────────────────────────────────────
    model = _load_pretrained(checkpoint_path, device)
    teacher = copy.deepcopy(model).to(device).eval()
    for p in teacher.parameters():
        p.requires_grad = False

    train_loader, test_loader, _ = get_cifar_loaders('cifar10', batch_size=128,
                                                      data_path=data_path, num_workers=4)

    # ── look up TG from scan file (or recompute) ─────────────────────────
    scan_file = _scan_path(save_dir)
    block_tg = None
    tg_rank  = None
    if os.path.exists(scan_file):
        with open(scan_file) as f:
            scan = json.load(f)
        for entry in scan['qualifying_blocks']:
            if entry['block'] == block_name:
                block_tg = entry['tg']
                tg_rank  = entry['tg_rank']
                break
        if block_tg is None:
            # Block not in scan — compute on the fly
            print(f"  {block_name} not in scan file; computing TG now …")
            tg_scores = compute_block_tg(model, train_loader, device,
                                         num_batches=CALIB_BATCHES)
            block_tg = tg_scores.get(block_name, float('nan'))
    else:
        print("Scan file not found — computing TG on the fly …")
        tg_scores = compute_block_tg(model, train_loader, device,
                                     num_batches=CALIB_BATCHES)
        block_tg = tg_scores.get(block_name, float('nan'))

    print(f"  TG={block_tg:.6f}  rank={tg_rank}")

    # ── verify block exists and is prunable ─────────────────────────────
    layer_name, idx_str = block_name.rsplit('.', 1)
    layer = getattr(model, layer_name, None)
    if layer is None or int(idx_str) >= len(layer):
        raise ValueError(f"Block {block_name} not found in ResNet-110")

    # ── remove block ─────────────────────────────────────────────────────
    model.replace_block_with_identity(block_name)
    print(f"  Replaced {block_name} with IdentityBlock")

    # ── 150-epoch fine-tune ───────────────────────────────────────────────
    ft_config = {
        'lr':              0.01,
        'momentum':        0.9,
        'weight_decay':    5e-4,
        'epochs':          FT_EPOCHS,
        'warmup_epochs':   5,
        'label_smoothing': 0.1,
        'use_mixup':       True,
        'mixup_alpha':     0.2,
        'use_kd':          True,
        'kd_alpha':        0.7,
        'kd_temperature':  4.0,
        'grad_clip':       5.0,
        'save_every':      50,
    }

    os.makedirs(save_dir, exist_ok=True)
    safe = block_name.replace('.', '_')
    ckpt_path = os.path.join(save_dir, f'e1_r110_{safe}_best.pth')

    trainer = Trainer(model, train_loader, test_loader, device, ft_config, teacher=teacher)
    result  = trainer.train(save_path=ckpt_path)
    final_acc = result['best_acc']

    acc_drop = round(BASELINE_ACC - final_acc, 3)
    print(f"\n  Δacc = {acc_drop:+.3f}%  (baseline {BASELINE_ACC:.2f}% → {final_acc:.2f}%)")

    summary = {
        'backbone':     BACKBONE,
        'block':        block_name,
        'tg':           round(block_tg, 6) if math.isfinite(block_tg) else None,
        'tg_rank':      tg_rank,
        'seed':         seed,
        'baseline_acc': BASELINE_ACC,
        'final_acc':    round(final_acc, 2),
        'actual_acc_drop': acc_drop,
        'ft_epochs':    FT_EPOCHS,
    }
    out = _summary_path(save_dir, block_name)
    with open(out, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary saved: {out}")


# ── mode: aggregate ─────────────────────────────────────────────────────────

def run_aggregate(save_dir: str):
    """Read all per-block summaries and compute Pearson r."""
    files = sorted(glob.glob(os.path.join(save_dir, 'e1_r110_*_summary.json')))
    if not files:
        print(f"No e1_r110_*_summary.json files found in {save_dir}")
        return

    entries = []
    for path in files:
        with open(path) as f:
            entries.append(json.load(f))

    # Sort by TG rank (ascending) for display
    entries.sort(key=lambda e: (e.get('tg_rank') or 999, e.get('tg') or 0))

    print(f"\n{'Block':<14} {'TG':>10} {'Rank':>6} {'Δacc':>8}")
    print('-' * 42)
    for e in entries:
        tg_str   = f"{e['tg']:.6f}" if e.get('tg') is not None else "N/A"
        rank_str = str(e['tg_rank']) if e.get('tg_rank') else "?"
        print(f"  {e['block']:<12} {tg_str:>10} {rank_str:>6} {e['actual_acc_drop']:>+8.3f}%")

    if len(entries) < 4:
        print("\nNeed at least 4 data points for a meaningful correlation.")
        return

    # Pearson r: TG rank vs actual_acc_drop
    ranks = [e['tg_rank'] for e in entries if e.get('tg_rank') is not None]
    drops = [e['actual_acc_drop'] for e in entries if e.get('tg_rank') is not None]
    n_rank = len(ranks)

    # Also: raw TG vs actual_acc_drop
    tg_vals  = [e['tg']           for e in entries if e.get('tg') is not None]
    drops_tg = [e['actual_acc_drop'] for e in entries if e.get('tg') is not None]
    n_tg = len(tg_vals)

    def pearson_r(xs, ys):
        n   = len(xs)
        mx  = sum(xs) / n
        my  = sum(ys) / n
        num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        dx  = math.sqrt(sum((x - mx) ** 2 for x in xs))
        dy  = math.sqrt(sum((y - my) ** 2 for y in ys))
        if dx < 1e-12 or dy < 1e-12:
            return float('nan'), float('nan'), float('nan')
        r   = num / (dx * dy)
        z   = 0.5 * math.log((1 + r) / (1 - r))
        se  = 1 / math.sqrt(n - 3)
        lo  = math.tanh(z - 1.96 * se)
        hi  = math.tanh(z + 1.96 * se)
        return r, lo, hi

    r_rank, lo_rank, hi_rank = pearson_r(ranks, drops)
    r_tg,   lo_tg,   hi_tg   = pearson_r(tg_vals, drops_tg)

    print(f"\n=== E1 Cross-Backbone Correlation (ResNet-110 / CIFAR-10) ===")
    print(f"  n blocks      = {n_rank}")
    print(f"  Pearson r (TG rank vs Δacc) = {r_rank:.4f}  95% CI [{lo_rank:.3f}, {hi_rank:.3f}]")
    print(f"  Pearson r (raw TG  vs Δacc) = {r_tg:.4f}   95% CI [{lo_tg:.3f}, {hi_tg:.3f}]")

    agg = {
        'backbone': BACKBONE,
        'baseline_acc': BASELINE_ACC,
        'tg_threshold': TG_THRESHOLD,
        'ft_epochs': FT_EPOCHS,
        'n': n_rank,
        'pearson_r_rank': round(r_rank, 4) if math.isfinite(r_rank) else None,
        'pearson_r_tg':   round(r_tg,   4) if math.isfinite(r_tg)   else None,
        'ci_rank': [round(lo_rank, 3), round(hi_rank, 3)],
        'ci_tg':   [round(lo_tg,   3), round(hi_tg,   3)],
        'per_block': entries,
    }
    out = os.path.join(save_dir, 'e1_r110_aggregate.json')
    with open(out, 'w') as f:
        json.dump(agg, f, indent=2)
    print(f"\n  Aggregate saved: {out}")


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='E1: Cross-backbone TG correlation (ResNet-110 / CIFAR-10)'
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--scan',      action='store_true',
                      help='Compute TG for all blocks, select TG < 0.025')
    mode.add_argument('--block',     type=str,
                      help='Remove this block and fine-tune (e.g. layer2.12)')
    mode.add_argument('--aggregate', action='store_true',
                      help='Compute Pearson r from all saved summaries')

    parser.add_argument('--checkpoint', type=str,
                        default='/arf/scratch/mhassan/transgap_v2/checkpoints/resnet110_cifar10_baseline.pth')
    parser.add_argument('--data_path',  type=str,
                        default='/arf/scratch/mhassan/cifar')
    parser.add_argument('--save_dir',   type=str,
                        default='/arf/scratch/mhassan/transgap_v2/checkpoints')
    parser.add_argument('--seed',       type=int, default=SEED)
    args = parser.parse_args()

    if args.scan:
        run_scan(args.checkpoint, args.data_path, args.save_dir)
    elif args.block:
        run_block(args.block, args.checkpoint, args.data_path,
                  args.save_dir, seed=args.seed)
    else:
        run_aggregate(args.save_dir)


if __name__ == '__main__':
    main()
