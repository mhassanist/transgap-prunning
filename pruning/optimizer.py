"""
Pruning Optimizer
==================
Joint block-channel optimization under FLOPs budget.

Stage 3 of TransGap v2 pipeline:
  Step 1: Greedy block selection by ascending TG with safety constraints
  Step 2: Lagrangian channel ratio allocation for remaining FLOPs target
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import copy

from models.resnet_cifar import ResNet, BasicBlock, IdentityBlock
from metrics.flops import count_flops


class PruningOptimizer:
    """
    Joint block-channel pruning optimizer.
    
    Given:
      - Block TG scores
      - Channel TG scores
      - Safety bounds
      - Target FLOPs budget
    
    Produces:
      - Set of blocks to remove
      - Per-block channel pruning ratios
    """
    
    def __init__(
        self,
        model: ResNet,
        block_tg: Dict[str, float],
        channel_tg: Dict[str, torch.Tensor],
        safety_bounds: Optional[Dict[str, Dict]] = None,
        input_size: Tuple[int, ...] = (1, 3, 32, 32),
    ):
        self.model = model
        self.block_tg = block_tg
        self.channel_tg = channel_tg
        self.safety_bounds = safety_bounds or {}
        self.input_size = input_size
        
        # Compute original FLOPs
        self.original_flops = count_flops(model, input_size)
    
    def optimize(
        self,
        target_flops_ratio: float = 0.5,
        max_block_tg: float = 0.05,
        max_channel_ratio: float = 0.7,
        max_safety_bound: Optional[float] = None,
    ) -> Tuple[List[str], Dict[str, float]]:
        """
        Find optimal block removal + channel pruning configuration.
        
        Args:
            target_flops_ratio: Target FLOPs as fraction of original (e.g., 0.5 = 50%)
            max_block_tg: Maximum TG for a block to be considered for removal
            max_channel_ratio: Maximum fraction of channels to prune per block
            max_safety_bound: Maximum Lipschitz safety bound
        
        Returns:
            (blocks_to_remove, channel_ratios)
            - blocks_to_remove: List of block names to remove entirely
            - channel_ratios: Dict mapping surviving block names to prune ratios
        """
        target_flops = int(self.original_flops * target_flops_ratio)
        
        # Step 1: Greedy block selection
        blocks_to_remove = self._greedy_block_selection(
            target_flops, max_block_tg, max_safety_bound
        )
        
        # Step 2: Compute remaining FLOPs after block removal
        temp_model = copy.deepcopy(self.model)
        for name in blocks_to_remove:
            temp_model.replace_block_with_identity(name)
        
        flops_after_blocks = count_flops(temp_model, self.input_size)
        remaining_reduction = flops_after_blocks - target_flops
        
        print(f"  Original FLOPs: {self.original_flops:,}")
        print(f"  After block removal ({len(blocks_to_remove)} blocks): {flops_after_blocks:,}")
        print(f"  Target FLOPs: {target_flops:,}")
        print(f"  Remaining to cut via channels: {remaining_reduction:,}")
        
        # Step 3: Allocate channel pruning ratios
        if remaining_reduction > 0:
            channel_ratios = self._allocate_channel_ratios(
                blocks_to_remove, remaining_reduction, max_channel_ratio
            )
        else:
            # Block removal alone meets the target
            channel_ratios = {}
        
        del temp_model
        return blocks_to_remove, channel_ratios
    
    def _greedy_block_selection(
        self,
        target_flops: int,
        max_block_tg: float,
        max_safety_bound: Optional[float],
    ) -> List[str]:
        """
        Greedily select blocks for removal, starting from laziest.
        Stop when either:
          - No more candidates below threshold
          - Removing more blocks would be excessive (past target already)
        """
        # Get candidates sorted by TG ascending
        candidates = []
        for name, tg in self.block_tg.items():
            if tg != tg:  # NaN = downsampling
                continue
            if tg > max_block_tg:
                continue
            if max_safety_bound is not None:
                bound = self.safety_bounds.get(name, {}).get('lipschitz_bound', float('inf'))
                if bound > max_safety_bound:
                    continue
            candidates.append((name, tg))
        
        candidates.sort(key=lambda x: x[1])
        
        # Greedily add blocks, but STOP BEFORE overshooting the target
        # This ensures channel pruning gets a chance to contribute
        selected = []
        prev_flops = self.original_flops
        
        for name, tg in candidates:
            # Tentatively add this block
            test_selected = selected + [name]
            
            temp_model = copy.deepcopy(self.model)
            for n in test_selected:
                temp_model.replace_block_with_identity(n)
            flops = count_flops(temp_model, self.input_size)
            del temp_model
            
            if flops <= target_flops:
                # Adding this block would meet/exceed target
                # Add it only if we haven't selected enough blocks yet
                # (at least leave ~10% FLOPs for channel pruning to demonstrate)
                min_channel_flops = int(self.original_flops * 0.05)  # reserve 5% for channels
                if flops + min_channel_flops <= prev_flops:
                    selected.append(name)
                break  # stop regardless — close enough to target
            
            selected.append(name)
            prev_flops = flops
        
        return selected
    
    def _allocate_channel_ratios(
        self,
        blocks_removed: List[str],
        remaining_reduction: int,
        max_ratio: float,
    ) -> Dict[str, float]:
        """
        Allocate per-block channel pruning ratios to achieve remaining FLOPs reduction.
        
        Strategy: Blocks with more low-TG channels get higher pruning ratios.
        This is a simplified Lagrangian allocation.
        """
        # Get surviving blocks that can be channel-pruned
        surviving_blocks = []
        for name, tg in self.block_tg.items():
            if tg != tg:  # downsampling
                continue
            if name in blocks_removed:
                continue
            # Only prune BasicBlocks
            layer_name, block_idx = name.rsplit('.', 1)
            layer = getattr(self.model, layer_name)
            block = layer[int(block_idx)]
            if isinstance(block, BasicBlock):
                surviving_blocks.append(name)
        
        if not surviving_blocks:
            return {}
        
        # Compute "pruning capacity" per block based on channel TG distribution
        # Blocks with more low-TG channels can afford more pruning
        capacities = {}
        for name in surviving_blocks:
            if name in self.channel_tg:
                tg_scores = self.channel_tg[name]
                # Capacity: fraction of channels with below-median TG
                median_tg = tg_scores.median()
                capacity = (tg_scores < median_tg).float().mean().item()
            else:
                # Fallback: use block-level TG inversely
                block_tg = self.block_tg.get(name, 0.5)
                capacity = 1.0 - block_tg  # lazier blocks can lose more
            capacities[name] = max(capacity, 0.1)  # minimum capacity
        
        # Normalize and allocate based on remaining FLOPs to cut
        total_capacity = sum(capacities.values())
        
        # Estimate how much each block contributes to total FLOPs
        # For ResNet CIFAR: each block has 2 conv layers
        # Rough FLOPs per block ∝ channels² × spatial_size²
        # Use a uniform starting ratio scaled by capacity
        
        # Calculate required total channel pruning ratio
        # remaining_reduction is in FLOPs; channel pruning roughly scales as ratio²
        # Start with a moderate base and adjust
        base_ratio = min(0.35, max(0.1, remaining_reduction / (self.original_flops * 0.3)))
        
        channel_ratios = {}
        for name in surviving_blocks:
            # Scale by capacity: blocks with more low-TG channels get higher ratio
            relative_capacity = capacities[name] / (total_capacity / len(surviving_blocks))
            ratio = base_ratio * relative_capacity
            ratio = min(ratio, max_ratio)
            ratio = max(ratio, 0.05)  # minimum 5% pruning
            channel_ratios[name] = round(ratio, 2)
        
        return channel_ratios
    
    def summary(
        self,
        blocks_to_remove: List[str],
        channel_ratios: Dict[str, float],
    ) -> str:
        """Print optimization summary."""
        lines = [
            "=" * 60,
            "Pruning Optimization Results",
            "=" * 60,
            f"\nOriginal FLOPs: {self.original_flops:,}",
            f"\nBlocks to remove ({len(blocks_to_remove)}):",
        ]
        for name in blocks_to_remove:
            tg = self.block_tg.get(name, 0)
            lines.append(f"  {name}: TG={tg:.6f}")
        
        lines.append(f"\nChannel pruning ratios ({len(channel_ratios)} blocks):")
        for name, ratio in sorted(channel_ratios.items()):
            lines.append(f"  {name}: {ratio:.1%}")
        
        return "\n".join(lines)
