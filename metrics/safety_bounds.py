"""
Safety Bound Estimation
========================
Theorem 1 (Lipschitz): ||Δoutput|| ≤ L_g · ||X_i|| · sqrt(2·TG)     [Eq. 11]
Theorem 2 (DPI):       ΔI_i ≈ -1/2 · log(1 - (1-ε)²)                [Eq. 15]

Provides computable guarantees for block removal safety.
"""

import torch
import torch.nn as nn
import math
from typing import Dict, List, Tuple
from tqdm import tqdm


def spectral_norm_estimate(weight: torch.Tensor, num_iters: int = 5) -> float:
    """
    Estimate spectral norm (largest singular value) of a weight matrix
    using power iteration. O(num_iters) matrix-vector multiplications.
    
    For conv layers, reshapes to (out_channels, in_channels * kH * kW).
    """
    if weight.dim() == 4:
        # Conv weight: (out_c, in_c, kH, kW) -> (out_c, in_c*kH*kW)
        W = weight.view(weight.size(0), -1)
    elif weight.dim() == 2:
        W = weight
    else:
        return 1.0  # fallback for BN etc.
    
    # Power iteration
    u = torch.randn(W.size(0), device=W.device)
    u = u / u.norm()
    
    for _ in range(num_iters):
        v = W.t() @ u
        v = v / (v.norm() + 1e-8)
        u = W @ v
        u = u / (u.norm() + 1e-8)
    
    sigma = (u @ W @ v).item()
    return abs(sigma)


def estimate_lipschitz(
    model: nn.Module,
    block_names: List[str],
) -> Dict[str, float]:
    """
    Estimate Lipschitz constant from each block to the output.
    
    L_{g_i} = product of spectral norms of all layers downstream of block i.
    
    This is an upper bound — the true Lipschitz constant may be smaller
    due to ReLU activations (which have Lipschitz constant 1).
    
    Args:
        model: The network
        block_names: Block names to compute Lipschitz constants for
    
    Returns:
        Dict mapping block_name -> L_g (Lipschitz constant to output)
    """
    # Collect all conv and linear layers in order
    layer_specs = []  # (name, spectral_norm)
    
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            sigma = spectral_norm_estimate(module.weight.data)
            layer_specs.append((name, sigma))
    
    # For each block, compute product of spectral norms of all layers AFTER it
    # We need to map block names to their position in the layer list
    lipschitz = {}
    
    for block_name in block_names:
        # Find the last layer within this block
        block_layer_idx = -1
        for idx, (name, _) in enumerate(layer_specs):
            if name.startswith(block_name):
                block_layer_idx = idx
        
        if block_layer_idx < 0:
            lipschitz[block_name] = float('inf')
            continue
        
        # Product of spectral norms of all layers after this block
        L_g = 1.0
        for idx in range(block_layer_idx + 1, len(layer_specs)):
            L_g *= layer_specs[idx][1]
        
        lipschitz[block_name] = L_g
    
    return lipschitz


@torch.no_grad()
def compute_safety_bounds(
    model: nn.Module,
    block_tg: Dict[str, float],
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    num_batches: int = 10,
) -> Dict[str, Dict[str, float]]:
    """
    Compute both Lipschitz and DPI safety bounds for each block.
    
    Returns dict: block_name -> {
        'tg': float,
        'lipschitz_bound': float,   # ||Δoutput|| upper bound
        'dpi_info_loss': float,     # ΔI (information loss)
        'lipschitz_constant': float, # L_g
        'input_norm': float,        # mean ||X_i||
    }
    """
    model.eval()
    model.to(device)
    
    # Filter to prunable blocks (non-NaN TG)
    prunable = {k: v for k, v in block_tg.items() if v == v}  # exclude NaN
    block_names = list(prunable.keys())
    
    # 1. Estimate Lipschitz constants
    lipschitz = estimate_lipschitz(model, block_names)
    
    # 2. Compute mean input norms via hooks
    input_norms = {}
    hooks = {}
    handles = []
    
    for name in block_names:
        norms = []
        
        def make_hook(store):
            def hook(module, inp, out):
                x = inp[0]
                norm = x.view(x.size(0), -1).norm(dim=1).mean().item()
                store.append(norm)
            return hook
        
        layer_name, block_idx = name.rsplit('.', 1)
        layer = getattr(model, layer_name)
        block = layer[int(block_idx)]
        
        hooks[name] = norms
        h = block.register_forward_hook(make_hook(norms))
        handles.append(h)
    
    # Run calibration
    batch_count = 0
    for images, _ in dataloader:
        if batch_count >= num_batches:
            break
        images = images.to(device)
        model(images)
        batch_count += 1
    
    for h in handles:
        h.remove()
    
    # Compute mean input norms
    for name in block_names:
        input_norms[name] = sum(hooks[name]) / len(hooks[name]) if hooks[name] else 1.0
    
    # 3. Compute bounds
    results = {}
    for name in block_names:
        tg = prunable[name]
        L_g = lipschitz.get(name, float('inf'))
        x_norm = input_norms.get(name, 1.0)
        
        # Theorem 1: Lipschitz bound
        # ||Δoutput|| ≤ L_g · ||X_i|| · sqrt(2·TG)
        lip_bound = L_g * x_norm * math.sqrt(2.0 * tg) if tg > 0 else 0.0
        
        # Theorem 2: DPI information loss
        # ΔI ≈ -1/2 · log(1 - (1-ε)²) where ε = TG
        rho = 1.0 - tg  # cosine similarity
        rho_sq = rho ** 2
        if rho_sq < 1.0:
            dpi_loss = -0.5 * math.log(1.0 - rho_sq)
        else:
            dpi_loss = float('inf')  # perfect correlation, zero loss
        
        results[name] = {
            'tg': tg,
            'lipschitz_bound': lip_bound,
            'dpi_info_loss': dpi_loss,
            'lipschitz_constant': L_g,
            'input_norm': x_norm,
        }
    
    return results
