"""
FLOPs and Parameter Counting
==============================
Counts multiply-accumulate operations (MACs) and parameters for ResNets.
"""

import torch
import torch.nn as nn
from typing import Tuple


def count_conv_flops(module: nn.Conv2d, input_size: Tuple[int, int]) -> int:
    """Count FLOPs for a single conv layer given input spatial size."""
    h_in, w_in = input_size
    h_out = (h_in + 2 * module.padding[0] - module.dilation[0] * (module.kernel_size[0] - 1) - 1) // module.stride[0] + 1
    w_out = (w_in + 2 * module.padding[1] - module.dilation[1] * (module.kernel_size[1] - 1) - 1) // module.stride[1] + 1
    
    flops_per_element = module.in_channels * module.kernel_size[0] * module.kernel_size[1]
    if module.groups > 1:
        flops_per_element = flops_per_element // module.groups
    
    total_flops = flops_per_element * module.out_channels * h_out * w_out
    return total_flops


def count_linear_flops(module: nn.Linear) -> int:
    """Count FLOPs for a linear layer."""
    return module.in_features * module.out_features


def count_flops(model: nn.Module, input_size: Tuple[int, ...] = (1, 3, 32, 32)) -> int:
    """
    Count total FLOPs for a model given input size.
    Uses hooks to track intermediate spatial dimensions.
    """
    total_flops = 0
    spatial_sizes = {}
    
    def hook_fn(name):
        def hook(module, input, output):
            nonlocal total_flops
            if isinstance(module, nn.Conv2d):
                h_in = input[0].shape[2]
                w_in = input[0].shape[3]
                flops = count_conv_flops(module, (h_in, w_in))
                total_flops += flops
            elif isinstance(module, nn.Linear):
                total_flops += count_linear_flops(module)
        return hook
    
    handles = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            h = module.register_forward_hook(hook_fn(name))
            handles.append(h)
    
    device = next(model.parameters()).device
    dummy = torch.randn(*input_size, device=device)
    
    model.eval()
    with torch.no_grad():
        model(dummy)
    
    for h in handles:
        h.remove()
    
    return total_flops


def count_params(model: nn.Module) -> int:
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_summary(model: nn.Module, input_size: Tuple[int, ...] = (1, 3, 32, 32)) -> str:
    """Print model FLOPs and parameter summary."""
    flops = count_flops(model, input_size)
    params = count_params(model)
    
    lines = [
        f"FLOPs:  {flops:>15,} ({flops/1e6:.2f}M)",
        f"Params: {params:>15,} ({params/1e6:.2f}M)",
    ]
    return "\n".join(lines)
