"""
Run ImageNet / ResNet-50 Pruning — Full Pipeline
==================================================
Matches the TransGap v2 pipeline diagram exactly:
  Stage 1: TG Computation (block + channel)
  Stage 2: Safety Bound Estimation (Lipschitz + DPI)
  Stage 3: Joint Block-Channel Optimization
  Stage 4a: Block Removal
  Stage 4b: Channel Pruning
  Stage 5: TG-Aware Fine-tuning (KD + TG-Scaled LR + Mixup)

Output files use 'full_' prefix to avoid conflicts with running experiments.

Usage:
    python -m experiments.run_imagenet_full --data_path /path/to/imagenet \
                                            --target_flops 0.5 --ft_epochs 90
"""

import argparse
import os
import sys
import json
import copy
import io
import contextlib
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights

from metrics.flops import count_flops, count_params
from metrics.safety_bounds import compute_safety_bounds
from training.trainer import Trainer
from utils.logger import setup_logger


# ============================================================
# Data Loading
# ============================================================

def get_imagenet_loaders(data_path, batch_size=64, num_workers=8):
    traindir = os.path.join(data_path, 'train')
    valdir = os.path.join(data_path, 'val')
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        normalize,
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    
    train_dataset = torchvision.datasets.ImageFolder(traindir, train_transform)
    val_dataset = torchvision.datasets.ImageFolder(valdir, val_transform)
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader


# ============================================================
# TG Computation
# ============================================================

def cosine_similarity_flat(x, y):
    x_flat = x.flatten(1)
    y_flat = y.flatten(1)
    return torch.nn.functional.cosine_similarity(x_flat, y_flat, dim=1)


@torch.no_grad()
def compute_resnet50_block_tg(model, dataloader, device, num_batches=10):
    """Block-level TG — memory efficient, incremental per block."""
    model.eval()
    model.to(device)
    
    block_info = []
    for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
        layer = getattr(model, layer_name)
        for block_idx in range(len(layer)):
            block_info.append((f"{layer_name}.{block_idx}", layer[block_idx]))
    
    tg_scores = {}
    
    for idx, (name, block) in enumerate(block_info):
        input_holder = []
        output_holder = []
        
        def make_hook(inp_list, out_list):
            def hook_fn(module, inp, out):
                inp_list.append(inp[0].detach())
                out_list.append(out.detach())
            return hook_fn
        
        handle = block.register_forward_hook(make_hook(input_holder, output_holder))
        
        running_cos_sum = 0.0
        total_samples = 0
        
        batch_count = 0
        for images, _ in dataloader:
            if batch_count >= num_batches:
                break
            images = images.to(device)
            model(images)
            
            inp = input_holder[-1]
            out = output_holder[-1]
            
            if inp.shape != out.shape:
                tg_scores[name] = float('nan')
                input_holder.clear()
                output_holder.clear()
                break
            
            cos = cosine_similarity_flat(inp, out)
            running_cos_sum += cos.sum().item()
            total_samples += cos.shape[0]
            input_holder.clear()
            output_holder.clear()
            batch_count += 1
        
        handle.remove()
        
        if name not in tg_scores and total_samples > 0:
            tg_scores[name] = 1.0 - running_cos_sum / total_samples
        
        if (idx + 1) % 4 == 0:
            print(f"  TG computed for {idx + 1}/{len(block_info)} blocks")
    
    return tg_scores


@torch.no_grad()
def compute_resnet50_channel_tg(model, dataloader, device, num_batches=5):
    """Channel-level importance via activation magnitude at conv2."""
    model.eval()
    model.to(device)
    
    channel_tg = {}
    
    for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
        layer = getattr(model, layer_name)
        for block_idx in range(len(layer)):
            block = layer[block_idx]
            name = f"{layer_name}.{block_idx}"
            
            if block_idx == 0 and hasattr(block, 'downsample') and block.downsample is not None:
                continue
            
            conv2 = block.conv2
            num_channels = conv2.out_channels
            output_holder = []
            
            def make_hook(out_list):
                def hook_fn(module, inp, out):
                    out_list.append(out.detach())
                return hook_fn
            
            handle = conv2.register_forward_hook(make_hook(output_holder))
            
            channel_importance = torch.zeros(num_channels, device=device)
            total_samples = 0
            
            batch_count = 0
            for images, _ in dataloader:
                if batch_count >= num_batches:
                    break
                images = images.to(device)
                model(images)
                
                out = output_holder[-1]
                importance = out.abs().mean(dim=[0, 2, 3])
                channel_importance += importance * out.shape[0]
                total_samples += out.shape[0]
                output_holder.clear()
                batch_count += 1
            
            handle.remove()
            
            if total_samples > 0:
                channel_importance /= total_samples
                if channel_importance.max() > 0:
                    channel_importance = channel_importance / channel_importance.max()
                channel_tg[name] = channel_importance.cpu().numpy()
    
    return channel_tg


# ============================================================
# Safety Bounds (Stage 2)
# ============================================================

@torch.no_grad()
def compute_resnet50_safety_bounds(model, block_tg, dataloader, device, num_batches=5):
    """Compute Lipschitz and DPI bounds for each non-downsample block."""
    model.eval()
    model.to(device)
    
    bounds = {}
    
    for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
        layer = getattr(model, layer_name)
        for block_idx in range(len(layer)):
            name = f"{layer_name}.{block_idx}"
            tg = block_tg.get(name, float('nan'))
            
            if tg != tg:  # NaN = downsample
                continue
            
            block = layer[block_idx]
            
            # Lipschitz bound: L = max spectral norm across convs
            lip = 0.0
            for m in block.modules():
                if isinstance(m, nn.Conv2d):
                    w = m.weight.data.flatten(1)
                    # Approximate spectral norm via power iteration
                    u = torch.randn(w.shape[0], device=device)
                    for _ in range(3):
                        v = torch.nn.functional.normalize(w.T @ u, dim=0)
                        u = torch.nn.functional.normalize(w @ v, dim=0)
                    sigma = (u @ w @ v).item()
                    lip = max(lip, sigma)
            
            # Bound: delta <= L * ||X|| * sqrt(2*TG)
            lip_bound = lip * math.sqrt(2.0 * tg) if tg > 0 else 0.0
            
            # DPI info loss: delta_I = -0.5 * log(1 - TG^2)
            tg_clamped = min(tg, 0.999)
            dpi_loss = -0.5 * math.log(1.0 - tg_clamped ** 2) if tg_clamped < 1.0 else float('inf')
            
            bounds[name] = {
                'tg': tg,
                'lipschitz': lip_bound,
                'dpi_loss': dpi_loss,
            }
    
    return bounds


# ============================================================
# Pruning Operations
# ============================================================

class IdentityBottleneck(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x


def prune_resnet50_blocks(model, blocks_to_remove):
    for name in blocks_to_remove:
        layer_name, block_idx = name.rsplit('.', 1)
        block_idx = int(block_idx)
        layer = getattr(model, layer_name)
        layer[block_idx] = IdentityBottleneck()
        print(f"  Removed block: {name}")
    return model


def prune_resnet50_channels(model, channel_prune_ratios, channel_tg):
    """Prune mid-channels (conv1 out, conv2 in+out, conv3 in) from Bottleneck blocks."""
    total_pruned = 0
    
    for name, ratio in channel_prune_ratios.items():
        if ratio <= 0:
            continue
        
        layer_name, block_idx = name.rsplit('.', 1)
        block_idx = int(block_idx)
        layer = getattr(model, layer_name)
        block = layer[block_idx]
        
        if isinstance(block, IdentityBottleneck):
            continue
        if name not in channel_tg:
            continue
        
        importance = channel_tg[name]
        num_channels = len(importance)
        num_to_prune = max(1, int(num_channels * ratio))
        num_to_keep = num_channels - num_to_prune
        
        if num_to_keep < 4:
            num_to_keep = 4
            num_to_prune = num_channels - num_to_keep
        
        keep_indices = np.argsort(importance)[-num_to_keep:]
        keep_indices = np.sort(keep_indices)
        keep_idx = torch.tensor(keep_indices, dtype=torch.long)
        
        # conv1 output
        old_conv1 = block.conv1
        new_conv1 = nn.Conv2d(old_conv1.in_channels, num_to_keep, kernel_size=1, bias=False)
        new_conv1.weight.data = old_conv1.weight.data[keep_idx].clone()
        block.conv1 = new_conv1
        
        # bn1
        old_bn1 = block.bn1
        new_bn1 = nn.BatchNorm2d(num_to_keep)
        new_bn1.weight.data = old_bn1.weight.data[keep_idx].clone()
        new_bn1.bias.data = old_bn1.bias.data[keep_idx].clone()
        new_bn1.running_mean = old_bn1.running_mean[keep_idx].clone()
        new_bn1.running_var = old_bn1.running_var[keep_idx].clone()
        block.bn1 = new_bn1
        
        # conv2 input+output
        old_conv2 = block.conv2
        new_conv2 = nn.Conv2d(num_to_keep, num_to_keep, kernel_size=3,
                              stride=old_conv2.stride, padding=1, groups=1, bias=False)
        new_conv2.weight.data = old_conv2.weight.data[keep_idx][:, keep_idx].clone()
        block.conv2 = new_conv2
        
        # bn2
        old_bn2 = block.bn2
        new_bn2 = nn.BatchNorm2d(num_to_keep)
        new_bn2.weight.data = old_bn2.weight.data[keep_idx].clone()
        new_bn2.bias.data = old_bn2.bias.data[keep_idx].clone()
        new_bn2.running_mean = old_bn2.running_mean[keep_idx].clone()
        new_bn2.running_var = old_bn2.running_var[keep_idx].clone()
        block.bn2 = new_bn2
        
        # conv3 input
        old_conv3 = block.conv3
        new_conv3 = nn.Conv2d(num_to_keep, old_conv3.out_channels, kernel_size=1, bias=False)
        new_conv3.weight.data = old_conv3.weight.data[:, keep_idx].clone()
        block.conv3 = new_conv3
        
        total_pruned += num_to_prune
        print(f"  {name}: pruned {num_to_prune}/{num_channels} mid-channels ({ratio*100:.0f}%)")
    
    return model, total_pruned


# ============================================================
# Main Pipeline
# ============================================================

def run_imagenet_full(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    data_path = config['data_path']
    batch_size = config.get('batch_size', 64)
    target_flops = config.get('target_flops_ratio', 0.5)
    max_blocks_to_remove = config.get('max_blocks', 4)
    
    save_dir = config.get('save_dir', './checkpoints')
    os.makedirs(save_dir, exist_ok=True)
    
    prefix = f"resnet50_imagenet_full_f{target_flops}"
    pruned_state_path = os.path.join(save_dir, f"{prefix}_pruned_state.pth")
    save_path = os.path.join(save_dir, f"{prefix}_best.pth")
    summary_path = os.path.join(save_dir, f"{prefix}_summary.json")
    
    if os.path.exists(summary_path):
        print(f"SKIP: Already completed — {summary_path}")
        with open(summary_path) as f:
            return json.load(f)
    
    # ========== Load Pretrained ==========
    print("\n" + "=" * 60)
    print("Loading Pretrained ResNet-50")
    print("=" * 60)
    
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    model.to(device)
    teacher = copy.deepcopy(model).eval()
    
    train_loader, val_loader = get_imagenet_loaders(data_path, batch_size)
    
    original_flops = count_flops(model, (1, 3, 224, 224))
    original_params = count_params(model)
    
    temp_trainer = Trainer(model, train_loader, val_loader, device, {'use_kd': False, 'use_mixup': False})
    baseline_acc = temp_trainer.evaluate()
    print(f"Baseline Top-1: {baseline_acc:.2f}%")
    print(f"FLOPs: {original_flops:,} ({original_flops/1e9:.2f}G)")
    print(f"Params: {original_params:,} ({original_params/1e6:.2f}M)")
    
    # Resume from pruned state if available
    if os.path.exists(pruned_state_path):
        print(f"\nResuming from pruned state: {pruned_state_path}")
        state = torch.load(pruned_state_path, map_location=device)
        model = state['model']
        model.to(device)
        block_tg = state['block_tg']
        blocks_removed = state['blocks_removed']
        channel_ratios_used = state.get('channel_ratios', {})
        safety_bounds = state.get('safety_bounds', {})
        pre_ft_acc = state.get('pre_ft_acc', 0.0)
        pruned_flops = count_flops(model, (1, 3, 224, 224))
        print(f"Loaded: {len(blocks_removed)} blocks removed, {(1-pruned_flops/original_flops)*100:.1f}% FLOPs reduction")
    else:
        # ========== Stage 1: TG Computation ==========
        print("\n" + "=" * 60)
        print("Stage 1: TG Computation (Block + Channel)")
        print("=" * 60)
        
        block_tg = compute_resnet50_block_tg(model, val_loader, device, num_batches=10)
        
        print(f"\n{'Block':<15} {'TG':>10} {'Status':>15}")
        print("-" * 40)
        for name, tg in sorted(block_tg.items()):
            if tg != tg: status = "DOWNSAMPLE"
            elif tg < 0.05: status = "LAZY"
            elif tg < 0.10: status = "MODERATE"
            elif tg < 0.15: status = "ACTIVE"
            else: status = "VERY ACTIVE"
            tg_str = f"{tg:.6f}" if tg == tg else "N/A"
            print(f"{name:<15} {tg_str:>10} {status:>15}")
        
        channel_tg = compute_resnet50_channel_tg(model, val_loader, device, num_batches=5)
        print(f"\nChannel-level TG computed for {len(channel_tg)} blocks")
        for name, scores in sorted(channel_tg.items()):
            print(f"  {name}: {len(scores)} ch, min={scores.min():.4f}, max={scores.max():.4f}, mean={scores.mean():.4f}")
        
        # ========== Stage 2: Safety Bound Estimation ==========
        print("\n" + "=" * 60)
        print("Stage 2: Safety Bound Estimation")
        print("=" * 60)
        
        safety_bounds = compute_resnet50_safety_bounds(model, block_tg, val_loader, device)
        
        print(f"\n{'Block':<15} {'TG':>10} {'Lipschitz':>12} {'DPI Loss':>12}")
        print("-" * 50)
        for name in sorted(safety_bounds.keys()):
            b = safety_bounds[name]
            print(f"{name:<15} {b['tg']:>10.6f} {b['lipschitz']:>12.4f} {b['dpi_loss']:>12.4f}")
        
        # ========== Stage 3: Joint Block-Channel Optimization ==========
        print("\n" + "=" * 60)
        print("Stage 3: Joint Block-Channel Optimization")
        print("=" * 60)
        
        candidates = [(n, tg) for n, tg in block_tg.items() if tg == tg]
        candidates.sort(key=lambda x: x[1])
        
        target_flops_val = int(original_flops * target_flops)
        blocks_removed = []
        current_flops = original_flops
        
        print(f"  Original FLOPs: {original_flops:,}")
        print(f"  Target FLOPs:   {target_flops_val:,}")
        
        for name, tg in candidates[:max_blocks_to_remove]:
            if current_flops <= target_flops_val:
                break
            temp = copy.deepcopy(model)
            test_blocks = blocks_removed + [name]
            with contextlib.redirect_stdout(io.StringIO()):
                prune_resnet50_blocks(temp, test_blocks)
            test_flops = count_flops(temp, (1, 3, 224, 224))
            del temp
            blocks_removed.append(name)
            current_flops = test_flops
            print(f"  Remove: {name} (TG={tg:.6f}), FLOPs: {current_flops/1e9:.2f}G ({(1-current_flops/original_flops)*100:.1f}% red.)")
        
        # Stage 4a: Block removal
        prune_resnet50_blocks(model, blocks_removed)
        flops_after_blocks = count_flops(model, (1, 3, 224, 224))
        
        print(f"\n  After {len(blocks_removed)} block removals: {flops_after_blocks/1e9:.2f}G ({(1-flops_after_blocks/original_flops)*100:.1f}% red.)")
        
        # Stage 4b: Channel pruning — iterative ratio search
        channel_ratios_used = {}
        if flops_after_blocks > target_flops_val:
            valid_tgs = [(n, t) for n, t in block_tg.items() if t == t]
            max_tg = max(t for _, t in valid_tgs)
            min_tg = min(t for _, t in valid_tgs)
            tg_range = max_tg - min_tg if max_tg > min_tg else 1.0
            
            for attempt_ratio in [x * 0.05 for x in range(2, 19)]:
                channel_ratios_used = {}
                for name in sorted(channel_tg.keys()):
                    layer_name, block_idx_str = name.rsplit('.', 1)
                    block_idx_int = int(block_idx_str)
                    layer = getattr(model, layer_name)
                    if isinstance(layer[block_idx_int], IdentityBottleneck):
                        continue
                    block_tg_val = block_tg.get(name, 0.1)
                    if block_tg_val != block_tg_val:
                        continue
                    tg_norm = (block_tg_val - min_tg) / tg_range
                    block_ratio = attempt_ratio * (1.3 - 0.6 * tg_norm)
                    block_ratio = min(0.65, max(0.05, block_ratio))
                    channel_ratios_used[name] = block_ratio
                
                test_model = copy.deepcopy(model)
                with contextlib.redirect_stdout(io.StringIO()):
                    prune_resnet50_channels(test_model, channel_ratios_used, channel_tg)
                test_flops = count_flops(test_model, (1, 3, 224, 224))
                del test_model
                print(f"  Ratio {attempt_ratio:.2f} -> {test_flops/1e9:.2f}G ({(1-test_flops/original_flops)*100:.1f}% red.)")
                if test_flops <= target_flops_val * 1.02:
                    break
            
            print(f"\n  Selected base ratio: {attempt_ratio:.2f}")
            print(f"  Channel pruning (TG-guided):")
            model, total_ch = prune_resnet50_channels(model, channel_ratios_used, channel_tg)
            print(f"\n  Total channels pruned: {total_ch}")
        
        # Pruned stats
        print("\n" + "=" * 60)
        print("Pruned Model Statistics")
        print("=" * 60)
        
        model.to(device)
        pruned_flops = count_flops(model, (1, 3, 224, 224))
        pruned_params = count_params(model)
        print(f"Pruned FLOPs: {pruned_flops:,} ({pruned_flops/1e9:.2f}G)")
        print(f"FLOPs reduction: {(1-pruned_flops/original_flops)*100:.1f}%")
        print(f"Pruned Params: {pruned_params:,} ({pruned_params/1e6:.2f}M)")
        print(f"Params reduction: {(1-pruned_params/original_params)*100:.1f}%")
        
        pre_ft_trainer = Trainer(model, train_loader, val_loader, device, {'use_kd': False, 'use_mixup': False})
        pre_ft_acc = pre_ft_trainer.evaluate()
        print(f"Pre-finetuning Top-1: {pre_ft_acc:.2f}%")
        
        torch.save({
            'model': model,
            'block_tg': block_tg,
            'blocks_removed': blocks_removed,
            'channel_ratios': channel_ratios_used,
            'safety_bounds': safety_bounds,
            'baseline_acc': baseline_acc,
            'pre_ft_acc': pre_ft_acc,
            'original_flops': original_flops,
            'pruned_flops': pruned_flops,
        }, pruned_state_path)
        print(f"Saved pruned state: {pruned_state_path}")
    
    # ========== Stage 5: TG-Aware Fine-tuning ==========
    print("\n" + "=" * 60)
    print("Stage 5: TG-Aware Fine-tuning (KD + TG-Scaled LR + Mixup)")
    print("=" * 60)
    
    # Build TG-scaled LR: surviving blocks get LR scaled by (1 + TG)
    # Higher TG blocks need more learning (they do more transformation)
    tg_lr_scale = {}
    for name, tg in block_tg.items():
        if tg != tg:  # skip NaN (downsample)
            continue
        if name in blocks_removed:
            continue
        # Check block still exists
        layer_name, block_idx_str = name.rsplit('.', 1)
        block_idx_int = int(block_idx_str)
        layer = getattr(model, layer_name)
        if isinstance(layer[block_idx_int], IdentityBottleneck):
            continue
        # Scale: higher TG -> higher LR (more to recover)
        # Lower TG -> lower LR (less was changed)
        scale = 0.5 + tg * 5.0  # TG=0.08 -> 0.9x, TG=0.16 -> 1.3x, TG=0.20 -> 1.5x
        tg_lr_scale[name] = round(scale, 3)
    
    print(f"  TG-Scaled LR for {len(tg_lr_scale)} blocks:")
    for name, scale in sorted(tg_lr_scale.items()):
        print(f"    {name}: scale={scale:.3f}")
    
    ft_config = {
        'lr': config.get('ft_lr', 0.001),
        'momentum': 0.9,
        'weight_decay': 1e-4,
        'epochs': config.get('ft_epochs', 90),
        'warmup_epochs': 5,
        'label_smoothing': 0.1,
        'use_mixup': True,           # ← Enabled (matches diagram)
        'mixup_alpha': 0.2,
        'use_kd': True,
        'kd_alpha': 0.5,
        'kd_temperature': 4.0,
        'grad_clip': 0,
        'save_every': 10,
        'tg_lr_scale': tg_lr_scale,  # ← TG-Scaled LR (matches diagram)
    }
    
    trainer = Trainer(model, train_loader, val_loader, device, ft_config, teacher=teacher)
    
    # Resume training if checkpoint exists
    resume_ckpt = save_path.replace('.pth', '_resume.pth')
    if os.path.exists(resume_ckpt):
        print(f"Resuming training from: {resume_ckpt}")
        trainer.resume_from(resume_ckpt)
    
    result = trainer.train(save_path=save_path)
    
    # ========== Summary ==========
    pruned_flops = count_flops(model, (1, 3, 224, 224))
    pruned_params = count_params(model)
    
    summary = {
        'model': 'resnet50',
        'dataset': 'imagenet',
        'pipeline': 'full',
        'baseline_acc': baseline_acc,
        'pre_finetune_acc': pre_ft_acc,
        'final_acc': result['best_acc'],
        'acc_delta': round(result['best_acc'] - baseline_acc, 2),
        'original_flops': original_flops,
        'pruned_flops': pruned_flops,
        'flops_reduction_pct': round((1 - pruned_flops / original_flops) * 100, 1),
        'original_params': original_params,
        'pruned_params': pruned_params,
        'params_reduction_pct': round((1 - pruned_params / original_params) * 100, 1),
        'blocks_removed': blocks_removed,
        'num_blocks_removed': len(blocks_removed),
        'channel_ratios': {k: round(v, 3) for k, v in channel_ratios_used.items()},
        'block_tg': {k: round(v, 6) if v == v else 'NaN' for k, v in block_tg.items()},
        'safety_bounds': {k: {kk: round(vv, 6) for kk, vv in v.items()} for k, v in safety_bounds.items()},
        'tg_lr_scale': tg_lr_scale,
        'training_time_hours': round(result['total_time'] / 3600, 2),
        'target_flops_ratio': target_flops,
        'ft_config': {
            'use_mixup': True,
            'use_kd': True,
            'tg_scaled_lr': True,
        },
    }
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'=' * 60}")
    print("IMAGENET EXPERIMENT COMPLETE (Full Pipeline)")
    print(f"{'=' * 60}")
    print(f"Baseline:     {baseline_acc:.2f}%")
    print(f"Final:        {result['best_acc']:.2f}%")
    print(f"Delta:        {result['best_acc'] - baseline_acc:+.2f}%")
    print(f"FLOPs:        {(1-pruned_flops/original_flops)*100:.1f}% reduction")
    print(f"Params:       {(1-pruned_params/original_params)*100:.1f}% reduction")
    print(f"Blocks:       {len(blocks_removed)} removed")
    print(f"Pipeline:     Full (Safety Bounds + TG-Scaled LR + Mixup)")
    print(f"Summary:      {summary_path}")
    
    return summary


def main():
    parser = argparse.ArgumentParser(description='TransGap v2 — ImageNet Full Pipeline')
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--target_flops', type=float, default=0.5)
    parser.add_argument('--max_blocks', type=int, default=4)
    parser.add_argument('--ft_epochs', type=int, default=90)
    parser.add_argument('--ft_lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--log_dir', type=str, default='./logs')
    args = parser.parse_args()
    
    config = {
        'data_path': args.data_path,
        'target_flops_ratio': args.target_flops,
        'max_blocks': args.max_blocks,
        'ft_epochs': args.ft_epochs,
        'ft_lr': args.ft_lr,
        'batch_size': args.batch_size,
        'save_dir': args.save_dir,
    }
    
    setup_logger(args.log_dir, "imagenet_resnet50_full")
    
    print(f"{'=' * 60}")
    print(f"ImageNet FULL Pipeline: target_flops={args.target_flops}, max_blocks={args.max_blocks}")
    print(f"{'=' * 60}")
    
    run_imagenet_full(config)


if __name__ == '__main__':
    main()
