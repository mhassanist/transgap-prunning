"""
ResNet-50 for ImageNet — thin wrapper around torchvision.
Provides consistent interface with resnet_cifar.py.
"""

import torch
import torch.nn as nn
from torchvision.models import resnet50 as tv_resnet50, ResNet50_Weights
from typing import List


def resnet50_imagenet(pretrained: bool = True) -> nn.Module:
    """Load torchvision ResNet-50, optionally pretrained on ImageNet."""
    if pretrained:
        model = tv_resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    else:
        model = tv_resnet50(weights=None)
    return model


def get_resnet50_block_names() -> List[str]:
    """Block names for torchvision ResNet-50."""
    names = []
    for layer_idx, count in [(1, 3), (2, 4), (3, 6), (4, 3)]:
        for block_idx in range(count):
            names.append(f"layer{layer_idx}.{block_idx}")
    return names


def get_resnet50_prunable_indices() -> List[int]:
    """
    Non-downsampling block indices for ResNet-50.
    First block of each layer group has a downsample projection.
    """
    indices = []
    offset = 0
    for count in [3, 4, 6, 3]:
        for i in range(count):
            if i > 0:  # skip first block (downsample)
                indices.append(offset + i)
            offset += 1  # this is intentionally inside
        # fix: offset should be accumulated correctly
    # Correct calculation:
    # layer1: blocks 0,1,2 -> prunable: 1,2
    # layer2: blocks 3,4,5,6 -> prunable: 4,5,6
    # layer3: blocks 7,8,9,10,11,12 -> prunable: 8,9,10,11,12
    # layer4: blocks 13,14,15 -> prunable: 14,15
    return [1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 14, 15]
