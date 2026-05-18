"""
Channel Pruner
===============
Prunes channels within surviving blocks based on channel-level TG.
Channels with lowest TG_channel contribute least to the block's transformation.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from models.resnet_cifar import ResNet, BasicBlock


def prune_conv_layer(
    conv: nn.Conv2d,
    bn: nn.BatchNorm2d,
    keep_indices: torch.Tensor,
    dim: str = 'output',
) -> Tuple[nn.Conv2d, nn.BatchNorm2d]:
    """
    Prune a conv+BN pair by removing channels.
    
    Args:
        conv: Convolutional layer
        bn: BatchNorm layer
        keep_indices: Indices of channels to keep
        dim: 'output' to prune output channels, 'input' to prune input channels
    
    Returns:
        (new_conv, new_bn) with reduced channels
    """
    keep_indices = keep_indices.to(conv.weight.device)
    
    if dim == 'output':
        # Prune output channels
        new_weight = conv.weight.data[keep_indices]
        new_conv = nn.Conv2d(
            conv.in_channels, len(keep_indices),
            conv.kernel_size, conv.stride, conv.padding,
            bias=conv.bias is not None, groups=conv.groups
        )
        new_conv.weight.data = new_weight
        if conv.bias is not None:
            new_conv.bias.data = conv.bias.data[keep_indices]
        
        # Prune BN
        new_bn = nn.BatchNorm2d(len(keep_indices))
        new_bn.weight.data = bn.weight.data[keep_indices]
        new_bn.bias.data = bn.bias.data[keep_indices]
        new_bn.running_mean = bn.running_mean[keep_indices]
        new_bn.running_var = bn.running_var[keep_indices]
        
    elif dim == 'input':
        # Prune input channels
        new_weight = conv.weight.data[:, keep_indices]
        new_conv = nn.Conv2d(
            len(keep_indices), conv.out_channels,
            conv.kernel_size, conv.stride, conv.padding,
            bias=conv.bias is not None, groups=conv.groups
        )
        new_conv.weight.data = new_weight
        if conv.bias is not None:
            new_conv.bias.data = conv.bias.data.clone()
        
        # BN unchanged (operates on output channels)
        new_bn = bn
    
    return new_conv, new_bn


class ChannelPruner:
    """
    Channel-level pruning using channel TG scores.
    
    For each block, removes channels with lowest TG_channel 
    (least contribution to the block's transformation).
    """
    
    def __init__(
        self,
        model: ResNet,
        channel_tg: Dict[str, torch.Tensor],
    ):
        self.model = model
        self.channel_tg = channel_tg
    
    def prune_block_channels(
        self,
        block_name: str,
        prune_ratio: float,
    ) -> int:
        """
        Prune channels from a specific block.
        
        For a BasicBlock with conv1(in->mid) + conv2(mid->out):
        - We prune the INTERNAL channels (mid dimension)
        - This means pruning conv1 outputs and conv2 inputs
        
        Args:
            block_name: e.g., 'layer1.3'
            prune_ratio: fraction of channels to remove (0.0 to 1.0)
        
        Returns:
            Number of channels removed
        """
        layer_name, block_idx = block_name.rsplit('.', 1)
        block_idx = int(block_idx)
        layer = getattr(self.model, layer_name)
        block = layer[block_idx]
        
        if not isinstance(block, BasicBlock):
            print(f"  Skipping {block_name}: not a BasicBlock (maybe already removed)")
            return 0
        
        # Get channel TG scores
        if block_name in self.channel_tg:
            tg_scores = self.channel_tg[block_name]
        else:
            # Fallback: use L1-norm of conv2 weights
            tg_scores = block.conv2.weight.data.abs().sum(dim=(1, 2, 3))
        
        num_channels = len(tg_scores)
        num_to_prune = int(num_channels * prune_ratio)
        num_to_keep = num_channels - num_to_prune
        
        if num_to_keep < 1:
            num_to_keep = 1
            num_to_prune = num_channels - 1
        
        # Select channels to keep (highest TG = most important)
        keep_indices = torch.argsort(tg_scores, descending=True)[:num_to_keep]
        keep_indices = keep_indices.sort().values  # maintain order
        
        # Prune conv1 outputs
        block.conv1, block.bn1 = prune_conv_layer(
            block.conv1, block.bn1, keep_indices, dim='output'
        )
        
        # Prune conv2 inputs
        block.conv2, block.bn2 = prune_conv_layer(
            block.conv2, block.bn2, keep_indices, dim='input'
        )
        
        # Update the block in the model
        layer[block_idx] = block
        
        return num_to_prune
    
    def prune_all_blocks(
        self,
        prune_ratios: Dict[str, float],
    ) -> Dict[str, int]:
        """
        Prune channels from multiple blocks.
        
        Args:
            prune_ratios: Dict mapping block_name -> prune_ratio
        
        Returns:
            Dict mapping block_name -> num_channels_removed
        """
        results = {}
        for block_name, ratio in prune_ratios.items():
            if ratio <= 0:
                continue
            removed = self.prune_block_channels(block_name, ratio)
            results[block_name] = removed
            print(f"  {block_name}: removed {removed} channels (ratio={ratio:.2f})")
        
        return results
