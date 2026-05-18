"""
Training Engine
================
Supports:
  - Standard CE training (baseline)
  - Knowledge Distillation (KD) with temperature scaling
  - TG-scaled learning rates (per-block adaptive LR)
  - Mixup and CutMix augmentation
  - Cosine annealing with warmup
  - Label smoothing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Optional, Callable
import time
import math


class Trainer:
    """
    Unified training engine for baseline and fine-tuning.
    
    Usage:
        trainer = Trainer(model, train_loader, test_loader, device, config)
        trainer.train(num_epochs=300)
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        device: torch.device,
        config: dict,
        teacher: Optional[nn.Module] = None,
        logger: Optional[Callable] = None,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.config = config
        self.teacher = teacher.to(device).eval() if teacher is not None else None
        self.log = logger or print
        
        # Training config with defaults
        self.lr = config.get('lr', 0.1)
        self.momentum = config.get('momentum', 0.9)
        self.weight_decay = config.get('weight_decay', 5e-4)
        self.epochs = config.get('epochs', 300)
        self.warmup_epochs = config.get('warmup_epochs', 5)
        self.label_smoothing = config.get('label_smoothing', 0.1)
        
        # KD config
        self.kd_alpha = config.get('kd_alpha', 0.7)
        self.kd_temperature = config.get('kd_temperature', 4.0)
        self.use_kd = teacher is not None and config.get('use_kd', True)
        
        # Mixup config
        self.mixup_alpha = config.get('mixup_alpha', 0.2)
        self.use_mixup = config.get('use_mixup', True)
        
        # TG-scaled LR config
        self.tg_lr_scale = config.get('tg_lr_scale', None)  # Dict[param_group_name -> scale]
        
        # Setup optimizer and scheduler
        self.optimizer = self._build_optimizer()
        self.scheduler = None  # built in train()
        
        # Resume support
        self.start_epoch = 0
        self.save_every = config.get('save_every', 50)  # checkpoint every N epochs
        
        # Tracking
        self.best_acc = 0.0
        self.history = {'train_loss': [], 'train_acc': [], 'test_acc': [], 'lr': []}
    
    def _build_optimizer(self) -> optim.Optimizer:
        """Build optimizer, optionally with per-block LR scaling."""
        if self.tg_lr_scale is not None:
            # Group parameters by block for TG-scaled LR
            param_groups = []
            assigned_params = set()
            
            for group_name, scale in self.tg_lr_scale.items():
                params = []
                for name, param in self.model.named_parameters():
                    # Exact block match: "layer1.5." matches "layer1.5.conv1.weight"
                    # but NOT "layer1.15.conv1.weight"
                    if name.startswith(group_name + '.') and param.requires_grad:
                        if name not in assigned_params:
                            params.append(param)
                            assigned_params.add(name)
                if params:
                    param_groups.append({
                        'params': params,
                        'lr': self.lr * scale,
                        'name': group_name,
                    })
            
            # Remaining params at base LR
            remaining = [
                p for n, p in self.model.named_parameters()
                if n not in assigned_params and p.requires_grad
            ]
            if remaining:
                param_groups.append({
                    'params': remaining,
                    'lr': self.lr,
                    'name': 'base',
                })
            
            return optim.SGD(
                param_groups,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
            )
        else:
            return optim.SGD(
                self.model.parameters(),
                lr=self.lr,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
            )
    
    def _get_lr(self, epoch: int) -> float:
        """Cosine annealing with linear warmup."""
        if epoch < self.warmup_epochs:
            return self.lr * (epoch + 1) / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / (self.epochs - self.warmup_epochs)
            return self.lr * 0.5 * (1 + math.cos(math.pi * progress))
    
    def _adjust_lr(self, epoch: int):
        """Apply learning rate schedule."""
        new_lr = self._get_lr(epoch)
        for param_group in self.optimizer.param_groups:
            if self.tg_lr_scale is not None and 'name' in param_group:
                scale = self.tg_lr_scale.get(param_group['name'], 1.0)
                param_group['lr'] = new_lr * scale
            else:
                param_group['lr'] = new_lr
    
    def _mixup_data(self, x, y):
        """Apply mixup augmentation."""
        if not self.use_mixup or self.mixup_alpha <= 0:
            return x, y, y, 1.0
        
        lam = torch.distributions.Beta(self.mixup_alpha, self.mixup_alpha).sample().item()
        batch_size = x.size(0)
        index = torch.randperm(batch_size, device=x.device)
        
        mixed_x = lam * x + (1 - lam) * x[index]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam
    
    def _compute_loss(self, outputs, targets_a, targets_b, lam):
        """Compute loss with label smoothing and optional KD."""
        # CE loss with label smoothing
        ce_loss = lam * F.cross_entropy(outputs, targets_a, label_smoothing=self.label_smoothing) \
                + (1 - lam) * F.cross_entropy(outputs, targets_b, label_smoothing=self.label_smoothing)
        
        if not self.use_kd or self.teacher is None:
            return ce_loss
        
        # KD loss
        with torch.no_grad():
            # For mixup, we use targets_a's input (approximate)
            teacher_outputs = self.teacher(self._current_input)
        
        T = self.kd_temperature
        kd_loss = F.kl_div(
            F.log_softmax(outputs / T, dim=1),
            F.softmax(teacher_outputs / T, dim=1),
            reduction='batchmean'
        ) * (T * T)
        
        alpha = self.kd_alpha
        total_loss = (1 - alpha) * ce_loss + alpha * kd_loss
        return total_loss
    
    def train_one_epoch(self, epoch: int) -> tuple:
        """Train for one epoch. Returns (avg_loss, accuracy)."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images, labels = images.to(self.device), labels.to(self.device)
            self._current_input = images  # store for KD
            
            # Mixup
            mixed_images, targets_a, targets_b, lam = self._mixup_data(images, labels)
            
            # Forward
            outputs = self.model(mixed_images)
            loss = self._compute_loss(outputs, targets_a, targets_b, lam)
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            max_norm = self.config.get('grad_clip', 0)
            if max_norm > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
            
            self.optimizer.step()
            
            # Track stats (use original labels for accuracy)
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (lam * predicted.eq(targets_a).sum().item()
                       + (1 - lam) * predicted.eq(targets_b).sum().item())
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100.0 * correct / total
        return avg_loss, accuracy
    
    @torch.no_grad()
    def evaluate(self) -> float:
        """Evaluate on test set. Returns accuracy."""
        self.model.eval()
        correct = 0
        total = 0
        
        for images, labels in self.test_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            outputs = self.model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        return 100.0 * correct / total
    
    def resume_from(self, checkpoint_path: str):
        """
        Resume training from a checkpoint.
        Loads model weights, optimizer state, epoch, and best accuracy.
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if checkpoint.get('optimizer_state_dict'):
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.start_epoch = checkpoint.get('epoch', 0) + 1
        self.best_acc = checkpoint.get('best_acc', 0.0)
        if checkpoint.get('history'):
            self.history = checkpoint['history']
        self.log(f"Resumed from epoch {self.start_epoch}, best_acc={self.best_acc:.2f}%")
    
    def train(self, save_path: Optional[str] = None) -> Dict:
        """
        Full training loop with resume support and periodic checkpointing.
        
        Returns:
            Dict with training history and best accuracy.
        """
        self.log(f"Training: epochs {self.start_epoch}-{self.epochs}, LR={self.lr}, "
                f"KD={'ON' if self.use_kd else 'OFF'}, "
                f"Mixup={'ON' if self.use_mixup else 'OFF'}")
        
        start_time = time.time()
        
        for epoch in range(self.start_epoch, self.epochs):
            self._adjust_lr(epoch)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Train
            train_loss, train_acc = self.train_one_epoch(epoch)
            
            # Evaluate
            test_acc = self.evaluate()
            
            # Track
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['test_acc'].append(test_acc)
            self.history['lr'].append(current_lr)
            
            # Best model
            if test_acc > self.best_acc:
                self.best_acc = test_acc
                if save_path:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'best_acc': self.best_acc,
                        'config': self.config,
                        'history': self.history,
                    }, save_path)
            
            # Periodic checkpoint for resume (every N epochs)
            if save_path and self.save_every > 0 and (epoch + 1) % self.save_every == 0:
                resume_path = save_path.replace('.pth', '_resume.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_acc': self.best_acc,
                    'config': self.config,
                    'history': self.history,
                }, resume_path)
                self.log(f"  [Checkpoint saved at epoch {epoch}]")
            
            # Log every 10 epochs or at key points
            if epoch % 10 == 0 or epoch == self.epochs - 1 or test_acc >= self.best_acc:
                elapsed = time.time() - start_time
                self.log(
                    f"Epoch {epoch:>3d}/{self.epochs} | "
                    f"LR {current_lr:.6f} | "
                    f"Loss {train_loss:.4f} | "
                    f"Train {train_acc:.2f}% | "
                    f"Test {test_acc:.2f}% | "
                    f"Best {self.best_acc:.2f}% | "
                    f"{elapsed/60:.1f}min"
                )
        
        total_time = time.time() - start_time
        self.log(f"\nTraining complete: {total_time/3600:.2f}h, Best accuracy: {self.best_acc:.2f}%")
        
        return {
            'best_acc': self.best_acc,
            'history': self.history,
            'total_time': total_time,
        }
