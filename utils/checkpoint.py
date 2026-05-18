"""Checkpoint utilities."""
import torch
import os
from typing import Optional


def save_checkpoint(
    model, optimizer, epoch, best_acc, config, path,
    extra: Optional[dict] = None,
):
    """Save a training checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
        'best_acc': best_acc,
        'config': config,
    }
    if extra:
        state.update(extra)
    torch.save(state, path)


def load_checkpoint(path, model, optimizer=None, device='cpu'):
    """Load a training checkpoint."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer and checkpoint.get('optimizer_state_dict'):
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint
