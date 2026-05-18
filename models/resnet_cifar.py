"""
ResNet for CIFAR-10/100
========================
Standard ResNet-56 and ResNet-110 for 32x32 images.
Architecture: conv(16) -> layer1(16, n) -> layer2(32, n) -> layer3(64, n) -> fc
  ResNet-56:  n=9  (27 residual blocks, 56 layers total)
  ResNet-110: n=18 (54 residual blocks, 110 layers total)

Each "block" is a BasicBlock with two 3x3 convolutions + skip connection.
Downsampling happens at the first block of layer2 and layer3 (stride=2).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class BasicBlock(nn.Module):
    """Standard residual block: y = x + F(x) where F = conv-bn-relu-conv-bn"""
    
    expansion = 1
    
    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        # Shortcut projection when dimensions change
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes)
            )
    
    @property
    def is_downsample(self) -> bool:
        """Whether this block performs spatial downsampling."""
        return len(self.shortcut) > 0
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = F.relu(out)
        return out


class IdentityBlock(nn.Module):
    """
    Replaces a residual block with identity: y = x
    Used when removing a lazy block (TG ≈ 0).
    Only valid for non-downsampling blocks (same in/out dimensions).
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class ResNet(nn.Module):
    """
    ResNet for CIFAR.
    
    Attributes:
        layer1, layer2, layer3: nn.Sequential of BasicBlock or IdentityBlock
        block_list: flat list of all residual blocks for easy indexing
        block_names: human-readable names like 'layer1.0', 'layer2.3', etc.
    """
    
    def __init__(self, block_cls, num_blocks: List[int], num_classes: int = 10):
        super().__init__()
        self.in_planes = 16
        self.num_blocks_per_layer = num_blocks
        
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        
        self.layer1 = self._make_layer(block_cls, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block_cls, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block_cls, 64, num_blocks[2], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64 * block_cls.expansion, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _make_layer(self, block_cls, planes: int, num_blocks: int, stride: int):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block_cls(self.in_planes, planes, s))
            self.in_planes = planes * block_cls.expansion
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    @property
    def block_list(self) -> List[nn.Module]:
        """Flat list of all residual blocks."""
        blocks = []
        for layer in [self.layer1, self.layer2, self.layer3]:
            for block in layer:
                blocks.append(block)
        return blocks
    
    @property
    def block_names(self) -> List[str]:
        """Human-readable block names like 'layer1.0', 'layer2.3'."""
        names = []
        for layer_idx, layer in enumerate([self.layer1, self.layer2, self.layer3], 1):
            for block_idx in range(len(layer)):
                names.append(f"layer{layer_idx}.{block_idx}")
        return names
    
    def get_prunable_block_indices(self) -> List[int]:
        """
        Returns indices of blocks that CAN be removed (non-downsampling blocks).
        Downsampling blocks (first block of layer2 and layer3) are never removed.
        """
        indices = []
        blocks = self.block_list
        for i, block in enumerate(blocks):
            if isinstance(block, BasicBlock) and not block.is_downsample:
                indices.append(i)
        return indices
    
    def replace_block_with_identity(self, block_name: str):
        """
        Replace a specific block with an IdentityBlock.
        
        Args:
            block_name: e.g., 'layer2.7' -> layer2[7]
        """
        layer_name, block_idx = block_name.rsplit('.', 1)
        block_idx = int(block_idx)
        layer = getattr(self, layer_name)
        
        # Safety check: don't remove downsampling blocks
        if isinstance(layer[block_idx], BasicBlock) and layer[block_idx].is_downsample:
            raise ValueError(f"Cannot remove downsampling block {block_name}")
        
        layer[block_idx] = IdentityBlock()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def resnet56(num_classes: int = 10) -> ResNet:
    """ResNet-56 for CIFAR: 27 residual blocks (9 per layer group)."""
    return ResNet(BasicBlock, [9, 9, 9], num_classes=num_classes)


def resnet110(num_classes: int = 10) -> ResNet:
    """ResNet-110 for CIFAR: 54 residual blocks (18 per layer group)."""
    return ResNet(BasicBlock, [18, 18, 18], num_classes=num_classes)
