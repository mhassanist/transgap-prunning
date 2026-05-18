"""
Transformation Gap Computation
================================
Block-level TG:   TG_block(i) = 1 - cos(X_i, Y_i)          [Eq. 2-3]
Channel-level TG: TG_channel(i,c) ≈ ||f_c||·||w_c^next|| / ||Y_i||  [Eq. 6]

Both measure functional behavior: does removing this unit change the computation?
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm


def cosine_similarity_flat(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Cosine similarity between two tensors, flattened per sample in batch."""
    a_flat = a.view(a.size(0), -1)  # (B, D)
    b_flat = b.view(b.size(0), -1)  # (B, D)
    cos = F.cosine_similarity(a_flat, b_flat, dim=1)  # (B,)
    return cos


class BlockTGHook:
    """Forward hook that records block input and output for TG computation."""
    
    def __init__(self):
        self.inputs = []
        self.outputs = []
    
    def __call__(self, module, input_tensor, output_tensor):
        # input_tensor is a tuple; take the first element
        self.inputs.append(input_tensor[0].detach())
        self.outputs.append(output_tensor.detach())
    
    def reset(self):
        self.inputs.clear()
        self.outputs.clear()


@torch.no_grad()
def compute_block_tg(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    num_batches: int = 10,
) -> Dict[str, float]:
    """
    Compute block-level Transformation Gap for all residual blocks.
    
    TG_block(i) = 1 - mean[cos(X_i, Y_i)] over calibration samples.
    
    Args:
        model: Pretrained model (must have .block_list and .block_names attributes)
        dataloader: Calibration data loader
        device: CPU or CUDA device
        num_batches: Number of batches to use for calibration
    
    Returns:
        Dict mapping block_name -> TG score (float)
    """
    model.eval()
    model.to(device)
    
    blocks = model.block_list
    block_names = model.block_names
    
    # Register hooks on all blocks
    hooks = {}
    hook_handles = []
    for name, block in zip(block_names, blocks):
        hook = BlockTGHook()
        hooks[name] = hook
        handle = block.register_forward_hook(hook)
        hook_handles.append(handle)
    
    # Run calibration data through network
    batch_count = 0
    for images, _ in dataloader:
        if batch_count >= num_batches:
            break
        images = images.to(device)
        model(images)
        batch_count += 1
    
    # Compute TG per block
    tg_scores = {}
    for name, hook in hooks.items():
        if not hook.inputs or not hook.outputs:
            tg_scores[name] = float('nan')
            continue
        
        # Concatenate all batches
        all_inputs = torch.cat(hook.inputs, dim=0)   # (N, C, H, W)
        all_outputs = torch.cat(hook.outputs, dim=0)  # (N, C, H, W)
        
        # Handle downsampling blocks: input and output may have different shapes
        # In this case, we skip (or use projection) — these blocks are not prunable
        if all_inputs.shape != all_outputs.shape:
            tg_scores[name] = float('nan')  # Mark as non-prunable
        else:
            cos_sim = cosine_similarity_flat(all_inputs, all_outputs)  # (N,)
            tg = (1.0 - cos_sim).mean().item()
            tg_scores[name] = tg
    
    # Clean up hooks
    for handle in hook_handles:
        handle.remove()
    for hook in hooks.values():
        hook.reset()
    
    return tg_scores


@torch.no_grad()
def compute_channel_tg(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    block_name: str,
    num_batches: int = 10,
    method: str = "approx",
) -> torch.Tensor:
    """
    Compute channel-level TG for all channels in a specific block.
    
    Approximate method (Eq. 6):
        TG_channel(i,c) ≈ ||f_c||_F · ||w_c^next||_F / ||Y_i||_F
    
    Exact method (Eq. 4):
        TG_channel(i,c) = 1 - cos(Y_i, Y_i^{\\c})  (masking each channel)
    
    Args:
        model: Pretrained model
        dataloader: Calibration data
        device: CPU or CUDA
        block_name: e.g., 'layer1.3'
        num_batches: Calibration batches
        method: 'approx' (Eq. 6, fast) or 'exact' (Eq. 4, slower)
    
    Returns:
        Tensor of shape (C,) with TG score per channel
    """
    model.eval()
    model.to(device)
    
    layer_name, block_idx = block_name.rsplit('.', 1)
    block_idx = int(block_idx)
    layer = getattr(model, layer_name)
    block = layer[block_idx]
    
    if method == "approx":
        return _channel_tg_approx(model, block, block_name, dataloader, device, num_batches)
    elif method == "exact":
        return _channel_tg_exact(model, block, block_name, dataloader, device, num_batches)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'approx' or 'exact'.")


def _channel_tg_approx(
    model: nn.Module,
    block: nn.Module,
    block_name: str,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    num_batches: int,
) -> torch.Tensor:
    """
    Approximate channel TG using activation magnitude × downstream weight magnitude.
    TG_channel(i,c) ≈ ||f_c||_F · ||w_c^next||_F / ||Y_i||_F
    
    Requires only pre-computed activations and weights — no extra passes.
    """
    # Get the last conv layer in the block (conv2 for BasicBlock)
    conv_last = block.conv2 if hasattr(block, 'conv2') else None
    if conv_last is None:
        raise ValueError(f"Block {block_name} has no conv2 layer")
    
    num_channels = conv_last.weight.shape[0]  # output channels of conv2
    
    # Get downstream weights: the next layer's input weights for this channel
    # For a residual block, the "next" consumer is the next block's conv1
    # We use the conv2 weight norms as a proxy for channel importance
    # Combined with activation norms
    
    # Hook to capture conv2 output (pre-BN, pre-addition)
    activations = []
    block_outputs = []
    
    def conv2_hook(module, inp, out):
        activations.append(out.detach())
    
    def block_hook(module, inp, out):
        block_outputs.append(out.detach())
    
    h1 = conv_last.register_forward_hook(conv2_hook)
    
    # Find the block module in the layer and register hook
    layer_name, block_idx_str = block_name.rsplit('.', 1)
    layer = getattr(model, layer_name)
    h2 = layer[int(block_idx_str)].register_forward_hook(block_hook)
    
    # Run calibration
    batch_count = 0
    for images, _ in dataloader:
        if batch_count >= num_batches:
            break
        images = images.to(device)
        model(images)
        batch_count += 1
    
    h1.remove()
    h2.remove()
    
    # activations: list of (B, C, H, W) tensors from conv2
    all_acts = torch.cat(activations, dim=0)    # (N, C, H, W)
    all_outs = torch.cat(block_outputs, dim=0)  # (N, C, H, W)
    
    # Per-channel activation norm: ||f_c||_F averaged over samples
    # Shape: (C,)
    act_norms = all_acts.pow(2).sum(dim=(0, 2, 3)).sqrt() / all_acts.size(0)
    
    # Block output norm: ||Y_i||_F averaged over samples
    out_norm = all_outs.pow(2).sum(dim=(0, 1, 2, 3)).sqrt() / all_outs.size(0)
    
    # Downstream weight norm: ||w_c^next||_F
    # Find the next conv layer that takes these channels as input
    next_weight_norms = _get_downstream_weight_norms(model, block_name, num_channels)
    
    # TG_channel(c) = act_norm(c) * weight_norm(c) / out_norm
    channel_tg = (act_norms * next_weight_norms) / (out_norm + 1e-8)
    
    return channel_tg


def _get_downstream_weight_norms(
    model: nn.Module,
    block_name: str,
    num_channels: int,
) -> torch.Tensor:
    """
    Get the L2 norm of weights in the next layer that receive each channel.
    Falls back to uniform if no clear next layer is found.
    """
    layer_name, block_idx = block_name.rsplit('.', 1)
    block_idx = int(block_idx)
    layer = getattr(model, layer_name)
    
    # Try the next block's conv1
    if block_idx + 1 < len(layer):
        next_block = layer[block_idx + 1]
        if hasattr(next_block, 'conv1'):
            # conv1 weight shape: (out_c, in_c, kH, kW)
            # in_c should match num_channels
            w = next_block.conv1.weight.data
            if w.shape[1] == num_channels:
                # Per input-channel norm: sum over (out_c, kH, kW)
                return w.pow(2).sum(dim=(0, 2, 3)).sqrt()
    
    # Fallback: use ones (all channels weighted equally)
    device = next(model.parameters()).device
    return torch.ones(num_channels, device=device)


def _channel_tg_exact(
    model: nn.Module,
    block: nn.Module,
    block_name: str,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    num_batches: int,
) -> torch.Tensor:
    """
    Exact channel TG: mask each channel and measure output change.
    TG_channel(i,c) = 1 - cos(Y_i, Y_i^{\\c})
    
    Slower (C+1 forward passes per batch) but exact.
    """
    layer_name, block_idx_str = block_name.rsplit('.', 1)
    layer = getattr(model, layer_name)
    block_module = layer[int(block_idx_str)]
    
    conv_last = block_module.conv2 if hasattr(block_module, 'conv2') else None
    if conv_last is None:
        raise ValueError(f"Block {block_name} has no conv2")
    num_channels = conv_last.weight.shape[0]
    
    # Collect block outputs with and without each channel
    block_outputs_full = []
    
    # Hook on the block
    def block_hook(module, inp, out):
        block_outputs_full.append(out.detach())
    
    handle = block_module.register_forward_hook(block_hook)
    
    # 1. Full forward pass (no masking)
    batch_count = 0
    for images, _ in dataloader:
        if batch_count >= num_batches:
            break
        images = images.to(device)
        model(images)
        batch_count += 1
    
    handle.remove()
    full_output = torch.cat(block_outputs_full, dim=0)  # (N, C, H, W)
    
    # 2. For each channel, mask it and measure change
    channel_tg = torch.zeros(num_channels, device=device)
    
    original_weight = conv_last.weight.data.clone()
    original_bias = conv_last.bias.data.clone() if conv_last.bias is not None else None
    
    for c in tqdm(range(num_channels), desc=f"Channel TG ({block_name})", leave=False):
        # Zero out channel c in conv2
        conv_last.weight.data[c] = 0
        if conv_last.bias is not None:
            conv_last.bias.data[c] = 0
        
        masked_outputs = []
        masked_handle = block_module.register_forward_hook(
            lambda mod, inp, out, store=masked_outputs: store.append(out.detach())
        )
        
        batch_count = 0
        for images, _ in dataloader:
            if batch_count >= num_batches:
                break
            images = images.to(device)
            model(images)
            batch_count += 1
        
        masked_handle.remove()
        masked_output = torch.cat(masked_outputs, dim=0)
        
        # TG = 1 - cos(full, masked)
        cos = cosine_similarity_flat(full_output, masked_output)
        channel_tg[c] = (1.0 - cos).mean()
        
        # Restore weights
        conv_last.weight.data = original_weight.clone()
        if original_bias is not None:
            conv_last.bias.data = original_bias.clone()
    
    return channel_tg


class TGAnalyzer:
    """
    High-level interface for TG analysis.
    Computes both block-level and channel-level TG, stores results.
    """
    
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self.block_tg: Dict[str, float] = {}
        self.channel_tg: Dict[str, torch.Tensor] = {}
    
    def analyze_blocks(
        self,
        dataloader: torch.utils.data.DataLoader,
        num_batches: int = 10,
    ) -> Dict[str, float]:
        """Compute block-level TG for all blocks."""
        self.block_tg = compute_block_tg(
            self.model, dataloader, self.device, num_batches
        )
        return self.block_tg
    
    def analyze_channels(
        self,
        dataloader: torch.utils.data.DataLoader,
        block_names: Optional[List[str]] = None,
        num_batches: int = 10,
        method: str = "approx",
    ) -> Dict[str, torch.Tensor]:
        """
        Compute channel-level TG for specified blocks.
        If block_names is None, analyzes all non-downsampling blocks.
        """
        if block_names is None:
            block_names = [
                name for name, tg in self.block_tg.items()
                if not (tg != tg)  # filter NaN (downsampling blocks)
            ]
        
        for name in tqdm(block_names, desc="Channel TG analysis"):
            self.channel_tg[name] = compute_channel_tg(
                self.model, dataloader, self.device, name, num_batches, method
            )
        
        return self.channel_tg
    
    def get_prunable_blocks_ranked(self) -> List[Tuple[str, float]]:
        """Return prunable blocks sorted by TG ascending (laziest first)."""
        ranked = [
            (name, tg) for name, tg in self.block_tg.items()
            if not (tg != tg)  # exclude NaN (downsampling)
        ]
        ranked.sort(key=lambda x: x[1])
        return ranked
    
    def get_channel_ranking(self, block_name: str) -> torch.Tensor:
        """Return channel indices sorted by TG ascending (least important first)."""
        if block_name not in self.channel_tg:
            raise ValueError(f"No channel TG data for {block_name}")
        tg = self.channel_tg[block_name]
        return torch.argsort(tg)  # ascending: least important first
    
    def summary(self) -> str:
        """Print a summary of TG analysis."""
        lines = ["=" * 60, "TransGap Analysis Summary", "=" * 60]
        
        if self.block_tg:
            lines.append("\nBlock-Level TG Scores:")
            lines.append(f"{'Block':<15} {'TG':>10} {'Status':>15}")
            lines.append("-" * 40)
            for name, tg in sorted(self.block_tg.items()):
                if tg != tg:  # NaN
                    status = "DOWNSAMPLE"
                elif tg < 0.01:
                    status = "VERY LAZY"
                elif tg < 0.05:
                    status = "LAZY"
                elif tg < 0.15:
                    status = "MODERATE"
                else:
                    status = "ACTIVE"
                tg_str = f"{tg:.6f}" if tg == tg else "N/A"
                lines.append(f"{name:<15} {tg_str:>10} {status:>15}")
        
        if self.channel_tg:
            lines.append(f"\nChannel-Level TG computed for {len(self.channel_tg)} blocks")
            for name, tg in self.channel_tg.items():
                lines.append(f"  {name}: {len(tg)} channels, "
                           f"min={tg.min():.6f}, max={tg.max():.6f}, "
                           f"mean={tg.mean():.6f}")
        
        return "\n".join(lines)
