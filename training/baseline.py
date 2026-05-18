"""
Baseline Training
==================
Train ResNet-56/110 on CIFAR-10/100 from scratch.
Target: 94.30%+ for ResNet-56 on CIFAR-10.
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
import json

from models import resnet56, resnet110
from training.trainer import Trainer


def get_cifar_loaders(
    dataset: str = 'cifar10',
    batch_size: int = 128,
    data_path: str = './data',
    num_workers: int = 4,
) -> tuple:
    """Get CIFAR-10 or CIFAR-100 data loaders with standard augmentation."""
    
    if dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        DatasetClass = torchvision.datasets.CIFAR10
        num_classes = 10
    elif dataset == 'cifar100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
        DatasetClass = torchvision.datasets.CIFAR100
        num_classes = 100
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    
    trainset = DatasetClass(root=data_path, train=True, download=True, transform=train_transform)
    testset = DatasetClass(root=data_path, train=False, download=True, transform=test_transform)
    
    train_loader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    test_loader = DataLoader(
        testset, batch_size=100, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, test_loader, num_classes


def train_baseline(config: dict) -> dict:
    """
    Train a baseline model from scratch.
    
    Config keys:
        model: 'resnet56' or 'resnet110'
        dataset: 'cifar10' or 'cifar100'
        epochs: int (default 300)
        batch_size: int (default 128)
        lr: float (default 0.1)
        save_dir: str (default './checkpoints')
        data_path: str (default './data')
    """
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    dataset = config.get('dataset', 'cifar10')
    batch_size = config.get('batch_size', 128)
    data_path = config.get('data_path', './data')
    save_dir = config.get('save_dir', './checkpoints')
    
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(data_path, exist_ok=True)
    
    # Data
    train_loader, test_loader, num_classes = get_cifar_loaders(
        dataset, batch_size, data_path
    )
    print(f"Dataset: {dataset}, Classes: {num_classes}")
    
    # Model
    model_name = config.get('model', 'resnet56')
    if model_name == 'resnet56':
        model = resnet56(num_classes=num_classes)
    elif model_name == 'resnet110':
        model = resnet110(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model: {model_name}, Parameters: {param_count:,}")
    
    # Training config (no KD for baseline)
    train_config = {
        'lr': config.get('lr', 0.1),
        'momentum': config.get('momentum', 0.9),
        'weight_decay': config.get('weight_decay', 5e-4),
        'epochs': config.get('epochs', 300),
        'warmup_epochs': config.get('warmup_epochs', 5),
        'label_smoothing': config.get('label_smoothing', 0.1),
        'use_mixup': config.get('use_mixup', True),
        'mixup_alpha': config.get('mixup_alpha', 0.2),
        'use_kd': False,
        'grad_clip': config.get('grad_clip', 5.0),
    }
    
    save_path = os.path.join(save_dir, f"{model_name}_{dataset}_baseline.pth")
    resume_path = save_path.replace('.pth', '_resume.pth')
    
    # Train
    trainer = Trainer(model, train_loader, test_loader, device, train_config)
    
    # Resume if checkpoint exists
    if os.path.exists(resume_path):
        print(f"Resuming from: {resume_path}")
        trainer.resume_from(resume_path)
    elif os.path.exists(save_path):
        print(f"Best checkpoint exists: {save_path}")
        # Check if training completed
        ckpt = torch.load(save_path, map_location=device, weights_only=False)
        if ckpt.get('epoch', 0) >= train_config['epochs'] - 1:
            print(f"Training already complete! Best acc: {ckpt['best_acc']:.2f}%")
            return {'best_acc': ckpt['best_acc'], 'history': ckpt.get('history', {}), 'total_time': 0}
    
    result = trainer.train(save_path=save_path)
    
    # Save training history
    history_path = os.path.join(save_dir, f"{model_name}_{dataset}_baseline_history.json")
    with open(history_path, 'w') as f:
        json.dump({
            'config': config,
            'train_config': train_config,
            'best_acc': result['best_acc'],
            'total_time_hours': result['total_time'] / 3600,
            'history': {
                'train_loss': result['history']['train_loss'],
                'test_acc': result['history']['test_acc'],
            }
        }, f, indent=2)
    
    print(f"\nCheckpoint saved: {save_path}")
    print(f"History saved: {history_path}")
    
    return result
