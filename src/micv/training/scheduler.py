from __future__ import annotations

from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, LRScheduler, SequentialLR


def build_warmup_cosine_scheduler(
    optimizer: Optimizer,
    epochs: int,
    steps_per_epoch: int,
    warmup_epochs: int = 1,
    min_learning_rate: float = 1.0e-7,
) -> LRScheduler:
    total_steps = max(1, epochs * steps_per_epoch)
    warmup_steps = max(0, warmup_epochs * steps_per_epoch)
    cosine_steps = max(1, total_steps - warmup_steps)

    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=cosine_steps,
        eta_min=min_learning_rate,
    )
    if warmup_steps == 0:
        return cosine_scheduler

    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_steps)
    return SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps],
    )