"""
Block Pruner
=============
Removes entire residual blocks identified as "lazy" (low TG).
Replaces them with identity connections.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional
from models.resnet_cifar import ResNet, IdentityBlock


class BlockPruner:
    """
    Block-level pruning using Transformation Gap scores.
    
    Strategy: Remove blocks with lowest TG (laziest) first,
    subject to safety bound constraints and FLOPs budget.
    """
    
    def __init__(
        self,
        model: ResNet,
        block_tg: Dict[str, float],
        safety_bounds: Optional[Dict[str, Dict]] = None,
    ):
        self.model = model
        self.block_tg = block_tg
        self.safety_bounds = safety_bounds
    
    def select_blocks_to_remove(
        self,
        num_blocks: Optional[int] = None,
        max_tg_threshold: float = 0.05,
        max_safety_bound: Optional[float] = None,
        target_flops_ratio: Optional[float] = None,
    ) -> List[str]:
        """
        Select which blocks to remove.
        
        Selection criteria (applied in order):
        1. Only non-downsampling blocks (NaN TG excluded)
        2. TG below max_tg_threshold
        3. Safety bound below max_safety_bound (if safety_bounds provided)
        4. Either top-k by lowest TG, or greedily until FLOPs target met
        
        Args:
            num_blocks: Fixed number of blocks to remove (if specified)
            max_tg_threshold: Maximum TG for a block to be considered removable
            max_safety_bound: Maximum Lipschitz bound for safe removal
            target_flops_ratio: Target FLOPs remaining (e.g., 0.5 for 50%)
        
        Returns:
            List of block names to remove, sorted by TG ascending
        """
        # Get prunable candidates
        candidates = []
        for name, tg in self.block_tg.items():
            if tg != tg:  # NaN = downsampling, skip
                continue
            if tg > max_tg_threshold:
                continue
            if max_safety_bound is not None and self.safety_bounds:
                bound = self.safety_bounds.get(name, {}).get('lipschitz_bound', float('inf'))
                if bound > max_safety_bound:
                    continue
            candidates.append((name, tg))
        
        # Sort by TG ascending (laziest first)
        candidates.sort(key=lambda x: x[1])
        
        if num_blocks is not None:
            # Fixed number: take top-k laziest
            selected = [name for name, _ in candidates[:num_blocks]]
        else:
            # Take all candidates below threshold
            selected = [name for name, _ in candidates]
        
        return selected
    
    def remove_blocks(self, blocks_to_remove: List[str]) -> ResNet:
        """
        Remove specified blocks from the model by replacing with IdentityBlock.
        
        Args:
            blocks_to_remove: List of block names (e.g., ['layer2.7', 'layer3.2'])
        
        Returns:
            Modified model (in-place modification)
        """
        for block_name in blocks_to_remove:
            self.model.replace_block_with_identity(block_name)
            print(f"  Removed block: {block_name} (TG={self.block_tg.get(block_name, '?'):.6f})")
        
        return self.model
    
    def prune(
        self,
        num_blocks: Optional[int] = None,
        max_tg_threshold: float = 0.05,
        max_safety_bound: Optional[float] = None,
    ) -> Tuple[ResNet, List[str]]:
        """
        Full block pruning pipeline.
        
        Returns:
            (pruned_model, list_of_removed_block_names)
        """
        blocks_to_remove = self.select_blocks_to_remove(
            num_blocks=num_blocks,
            max_tg_threshold=max_tg_threshold,
            max_safety_bound=max_safety_bound,
        )
        
        if not blocks_to_remove:
            print("No blocks selected for removal.")
            return self.model, []
        
        print(f"Removing {len(blocks_to_remove)} blocks:")
        self.remove_blocks(blocks_to_remove)
        
        return self.model, blocks_to_remove
