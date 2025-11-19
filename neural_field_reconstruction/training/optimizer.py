"""
Optimizer and Learning Rate Scheduler Utilities
"""

import torch
from torch.optim import Optimizer, Adam, AdamW, SGD
from torch.optim.lr_scheduler import (
    _LRScheduler,
    ExponentialLR,
    CosineAnnealingLR,
    StepLR,
)
from typing import Dict, Any


def create_optimizer(
    model_parameters,
    optimizer_config: Dict[str, Any],
) -> Optimizer:
    """
    Create optimizer from configuration.

    Args:
        model_parameters: Model parameters to optimize
        optimizer_config: Configuration dictionary

    Returns:
        optimizer: PyTorch optimizer
    """
    optimizer_type = optimizer_config.get("type", "adam").lower()
    lr = optimizer_config.get("lr", 1e-2)
    weight_decay = optimizer_config.get("weight_decay", 0.0)

    if optimizer_type == "adam":
        optimizer = Adam(
            model_parameters,
            lr=lr,
            betas=optimizer_config.get("betas", (0.9, 0.99)),
            eps=optimizer_config.get("eps", 1e-15),
            weight_decay=weight_decay,
        )
    elif optimizer_type == "adamw":
        optimizer = AdamW(
            model_parameters,
            lr=lr,
            betas=optimizer_config.get("betas", (0.9, 0.99)),
            eps=optimizer_config.get("eps", 1e-15),
            weight_decay=weight_decay,
        )
    elif optimizer_type == "sgd":
        optimizer = SGD(
            model_parameters,
            lr=lr,
            momentum=optimizer_config.get("momentum", 0.9),
            weight_decay=weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")

    return optimizer


def create_scheduler(
    optimizer: Optimizer,
    scheduler_config: Dict[str, Any],
) -> _LRScheduler:
    """
    Create learning rate scheduler from configuration.

    Args:
        optimizer: PyTorch optimizer
        scheduler_config: Configuration dictionary

    Returns:
        scheduler: Learning rate scheduler
    """
    scheduler_type = scheduler_config.get("type", "exponential").lower()

    if scheduler_type == "exponential":
        scheduler = ExponentialLR(
            optimizer,
            gamma=scheduler_config.get("gamma", 0.1 ** (1 / 10000)),
        )
    elif scheduler_type == "cosine":
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=scheduler_config.get("T_max", 10000),
            eta_min=scheduler_config.get("eta_min", 1e-6),
        )
    elif scheduler_type == "step":
        scheduler = StepLR(
            optimizer,
            step_size=scheduler_config.get("step_size", 5000),
            gamma=scheduler_config.get("gamma", 0.5),
        )
    elif scheduler_type == "none":
        scheduler = None
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")

    return scheduler


class WarmupScheduler(_LRScheduler):
    """
    Learning rate scheduler with linear warmup.

    Args:
        optimizer: PyTorch optimizer
        warmup_steps: Number of warmup steps
        base_scheduler: Base scheduler to use after warmup
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        base_scheduler: _LRScheduler = None,
    ):
        self.warmup_steps = warmup_steps
        self.base_scheduler = base_scheduler
        self.current_step = 0
        super().__init__(optimizer)

    def get_lr(self):
        if self.current_step < self.warmup_steps:
            # Linear warmup
            warmup_factor = self.current_step / self.warmup_steps
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # Use base scheduler
            if self.base_scheduler is not None:
                return self.base_scheduler.get_last_lr()
            else:
                return self.base_lrs

    def step(self, epoch=None):
        self.current_step += 1
        if self.current_step >= self.warmup_steps and self.base_scheduler is not None:
            self.base_scheduler.step()
        super().step(epoch)
