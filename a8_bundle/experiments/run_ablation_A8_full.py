"""
Full A8: cross-backbone TG-vs-empirical-Δaccuracy correlation.

For each candidate block in {ResNet-56, ResNet-110, ResNet-50}, replace that
single block with identity, fine-tune with the manuscript's §4.4 recipe
(150 epochs CIFAR / 90 epochs ImageNet, KD α=0.5, T=4, Mixup, cosine LR),
measure Δaccuracy. The output
is a list of (backbone, block, tg, actual_acc_drop) entries that the paper's
A8 ablation can use for an honest Pearson r computation.

Selection: lowest-TG blocks per backbone (where the criterion is supposed to
identify removable blocks). We use the top-K lowest-TG blocks where K is set
per backbone to give ~10-12 entries each, for ~32 total.

Usage:
    python -m experiments.run_ablation_A8_full --backbone r56_cifar10 --block layer1.5
    python -m experiments.run_ablation_A8_full --aggregate
"""
import argparse
import json
import os
import sys
import copy
import random
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Block selection: lowest-TG blocks per backbone (from orthogonality JSON)
BLOCK_SELECTIONS = {
    'r56_cifar10': [
        # 10 lowest-TG blocks on R56 (from orthogonality_r56_cifar10.json)
        'layer1.8', 'layer2.6', 'layer2.8', 'layer1.7', 'layer2.5',
        'layer2.4', 'layer2.7', 'layer1.5', 'layer1.3', 'layer2.3',
    ],
    'r110_cifar10': [
        # 12 lowest-TG blocks on R110 (mix of layer1/2/3)
        'layer2.12', 'layer2.13', 'layer2.14', 'layer2.15', 'layer2.16',
        'layer2.17', 'layer1.5', 'layer2.5', 'layer1.9', 'layer1.6',
        'layer1.14', 'layer2.10',
    ],
    'r50_imagenet': [
        # 8 lowest-TG blocks on R50-ImageNet
        'layer1.1', 'layer1.2', 'layer2.3', 'layer2.2', 'layer3.3',
        'layer3.2', 'layer3.4', 'layer2.1',
    ],
}

# TG values per block (cached from orthogonality runs)
BLOCK_TG = {
    'r56_cifar10': {
        'layer1.8': 0.002837, 'layer2.6': 0.003993, 'layer2.8': 0.010957,
        'layer1.7': 0.012956, 'layer2.5': 0.014059, 'layer2.4': 0.014634,
        'layer2.7': 0.014835, 'layer1.5': 0.020840, 'layer1.3': 0.021402,
        'layer2.3': 0.021764,
    },
    'r110_cifar10': {
        'layer2.12': 0.000018, 'layer2.13': 0.000017, 'layer2.14': 0.000016,
        'layer2.15': 0.000016, 'layer2.16': 0.000015, 'layer2.17': 0.000015,
        'layer1.5': 0.000964, 'layer2.5': 0.002181, 'layer1.9': 0.004065,
        'layer1.6': 0.002534, 'layer1.14': 0.004153, 'layer2.10': 0.000019,
    },
    'r50_imagenet': {
        'layer1.1': 0.075958, 'layer1.2': 0.078192, 'layer2.3': 0.085988,
        'layer2.2': 0.106905, 'layer3.3': 0.106033, 'layer3.2': 0.113741,
        'layer3.4': 0.114093, 'layer2.1': 0.161713,
    },
}


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_cifar_single_block(backbone, block_name, save_dir, data_path, checkpoint_dir, seed=42):
    """Remove ONE block on CIFAR-10 R56 or R110, fine-tune with the §4.4 recipe (150 epochs)."""
    from models.resnet_cifar import resnet56, resnet110
    from training.trainer import Trainer
    from utils.checkpoint import load_checkpoint
    from training.baseline import get_cifar_loaders

    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_fn = resnet56 if backbone == 'r56_cifar10' else resnet110
    model = model_fn(num_classes=10)
    baseline_ckpt = f"resnet{'56' if backbone=='r56_cifar10' else '110'}_cifar10_baseline.pth"
    load_checkpoint(os.path.join(checkpoint_dir, baseline_ckpt), model, device=device)
    teacher = copy.deepcopy(model).to(device).eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # Measure baseline accuracy
    train_loader, test_loader, _ = get_cifar_loaders('cifar10', batch_size=128,
                                                      data_path=data_path, num_workers=4)
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    baseline_acc = 100.0 * correct / total
    print(f"Baseline {backbone}: {baseline_acc:.2f}%")

    # Remove the single block (replace with identity) — the same
    # model.replace_block_with_identity() used by run_e1_r110_correlation.py
    # (proven on TRUBA: 34 real result JSONs in results/) and the mechanism
    # the manuscript's §4.4/S4a describes ("replacing each selected block
    # with an identity mapping"). No zero-weight or BlockPruner workaround
    # needed — resnet56/resnet110 both expose this method (models/resnet_cifar.py).
    model.replace_block_with_identity(block_name)
    print(f"Replaced {block_name} with IdentityBlock")

    # Fine-tune with the manuscript's §4.4 recipe (150 epochs, KD α=0.5, T=4, Mixup, cosine LR) —
    # this is the recipe §5.7/A8 states was used to produce Table 7, verbatim-confirmed against
    # "TransGap_manuscript formatted.docx" (old-work/paper1-transgap/archive/00_2026_root_archive.zip).
    ft_config = {
        'lr': 0.01, 'momentum': 0.9, 'weight_decay': 5e-4,
        'epochs': 150, 'warmup_epochs': 5,
        'label_smoothing': 0.1, 'use_mixup': True, 'mixup_alpha': 0.2,
        'use_kd': True, 'kd_alpha': 0.5, 'kd_temperature': 4.0, 'grad_clip': 5.0,
    }
    trainer = Trainer(model, train_loader, test_loader, device, ft_config, teacher=teacher)
    save_path = os.path.join(save_dir, f"a8full_{backbone}_{block_name}_seed{seed}.pth")
    result = trainer.train(save_path=save_path)
    final_acc = result['best_acc']

    summary = {
        'backbone': backbone,
        'block': block_name,
        'seed': seed,
        'tg': BLOCK_TG[backbone][block_name],
        'baseline_acc': round(baseline_acc, 2),
        'final_acc': round(final_acc, 2),
        'actual_acc_drop': round(baseline_acc - final_acc, 3),
        'ft_epochs': 150,
    }
    out_path = os.path.join(save_dir, f"a8full_{backbone}_{block_name}_summary.json")
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {out_path}")
    print(f"  Δacc = {summary['actual_acc_drop']:+.3f}")


def run_imagenet_single_block(block_name, save_dir, data_path, checkpoint_dir, seed=42):
    """Remove ONE block on ImageNet R50, fine-tune with the §4.4 recipe (90 epochs)."""
    import torchvision.models as tvm
    from training.trainer import Trainer
    from experiments.run_imagenet import get_imagenet_loaders, IdentityBottleneck, prune_resnet50_blocks

    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V1)
    teacher = copy.deepcopy(model).to(device).eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # Baseline already known: 76.13%
    baseline_acc = 76.13

    # Remove the single block
    model = prune_resnet50_blocks(model, [block_name])
    model.to(device)
    print(f"Removed block {block_name}")

    train_loader, val_loader = get_imagenet_loaders(data_path, batch_size=64, num_workers=4)
    ft_config = {
        'lr': 0.001, 'momentum': 0.9, 'weight_decay': 1e-4,
        'epochs': 90, 'warmup_epochs': 5,
        'label_smoothing': 0.1, 'use_mixup': False,
        'use_kd': True, 'kd_alpha': 0.5, 'kd_temperature': 4.0, 'grad_clip': 0,
    }
    trainer = Trainer(model, train_loader, val_loader, device, ft_config, teacher=teacher)
    save_path = os.path.join(save_dir, f"a8full_r50_imagenet_{block_name}_seed{seed}.pth")
    result = trainer.train(save_path=save_path)
    final_acc = result['best_acc']

    summary = {
        'backbone': 'r50_imagenet',
        'block': block_name,
        'seed': seed,
        'tg': BLOCK_TG['r50_imagenet'][block_name],
        'baseline_acc': baseline_acc,
        'final_acc': round(final_acc, 2),
        'actual_acc_drop': round(baseline_acc - final_acc, 3),
        'ft_epochs': 90,
    }
    out_path = os.path.join(save_dir, f"a8full_r50_imagenet_{block_name}_summary.json")
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {out_path}")
    print(f"  Δacc = {summary['actual_acc_drop']:+.3f}")


def aggregate(save_dir):
    """Read all a8full_*_summary.json files and compute Pearson r."""
    import glob, math
    files = sorted(glob.glob(os.path.join(save_dir, "a8full_*_summary.json")))
    print(f"Found {len(files)} A8 result files")
    entries = []
    for f in files:
        with open(f) as fh:
            d = json.load(fh)
        entries.append(d)
    print(f"\n{'Backbone':<15} {'Block':<14} {'TG':>10} {'Δacc':>8}")
    print('-' * 50)
    for e in entries:
        print(f"{e['backbone']:<15} {e['block']:<14} {e['tg']:>10.5f} {e['actual_acc_drop']:>+8.3f}")

    if len(entries) < 4:
        print("\nToo few entries for correlation")
        return

    tg = [e['tg'] for e in entries]
    ad = [e['actual_acc_drop'] for e in entries]
    n = len(tg)
    mx, my = sum(tg)/n, sum(ad)/n
    num = sum((t-mx)*(a-my) for t, a in zip(tg, ad))
    denx = math.sqrt(sum((t-mx)**2 for t in tg))
    deny = math.sqrt(sum((a-my)**2 for a in ad))
    r = num / (denx * deny)
    z = 0.5 * math.log((1+r)/(1-r))
    se = 1/math.sqrt(n-3)
    lo = math.tanh(z - 1.96*se); hi = math.tanh(z + 1.96*se)

    print(f"\n=== A8 Correlation ===")
    print(f"  n = {n}")
    print(f"  Pearson r = {r:.4f}")
    print(f"  95% CI (Fisher z) = [{lo:.3f}, {hi:.3f}]")

    # Save aggregate
    agg = {
        'n': n,
        'pearson_r': round(r, 4),
        'ci_low': round(lo, 3),
        'ci_high': round(hi, 3),
        'per_block': entries,
    }
    out = os.path.join(save_dir, 'ablation_A8_full_summary.json')
    with open(out, 'w') as f:
        json.dump(agg, f, indent=2)
    print(f"\nAggregate saved: {out}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', choices=['r56_cifar10', 'r110_cifar10', 'r50_imagenet'])
    parser.add_argument('--block', help="Block name like layer1.5")
    parser.add_argument('--aggregate', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_dir', default='/arf/scratch/mhassan/transgap_v2/checkpoints')
    parser.add_argument('--data_path_cifar', default='/arf/scratch/mhassan/cifar')
    parser.add_argument('--data_path_imagenet', default='/arf/scratch/mhassan/imagenet')
    parser.add_argument('--checkpoint_dir', default='/arf/scratch/mhassan/transgap_v2/checkpoints')
    args = parser.parse_args()

    if args.aggregate:
        aggregate(args.save_dir)
    elif args.backbone and args.block:
        if args.backbone == 'r50_imagenet':
            run_imagenet_single_block(args.block, args.save_dir,
                                       args.data_path_imagenet, args.checkpoint_dir, args.seed)
        else:
            run_cifar_single_block(args.backbone, args.block, args.save_dir,
                                    args.data_path_cifar, args.checkpoint_dir, args.seed)
    else:
        parser.error("Specify --backbone and --block, or --aggregate")
