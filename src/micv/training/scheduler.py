from __future__ import annotations

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, LRScheduler, SequentialLR


_NORM_MODULE_TYPES = tuple(
    module_type
    for module_type in (
        torch.nn.BatchNorm1d,
        torch.nn.BatchNorm2d,
        torch.nn.BatchNorm3d,
        torch.nn.SyncBatchNorm,
        torch.nn.InstanceNorm1d,
        torch.nn.InstanceNorm2d,
        torch.nn.InstanceNorm3d,
        torch.nn.LayerNorm,
        torch.nn.GroupNorm,
        getattr(torch.nn, "RMSNorm", None),
    )
    if module_type is not None
)


def build_param_groups(
    model: torch.nn.Module,
    backbone_lr: float,
    head_lr: float,
    weight_decay: float,
) -> list[dict]:
    decay_backbone, no_decay_backbone = [], []
    decay_head, no_decay_head = [], []
    modules = dict(model.named_modules())

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        is_backbone = (
            ".backbones." in name
            or name.startswith("stream1.backbones")
            or name.startswith("stream2.backbones")
        )
        module_name = name.rsplit(".", 1)[0] if "." in name else ""
        module = modules.get(module_name)
        no_decay = (
            param.ndim < 2
            or name.endswith(".bias")
            or isinstance(module, _NORM_MODULE_TYPES)
            or "norm" in name.lower()
        )

        if is_backbone and no_decay:
            no_decay_backbone.append(param)
        elif is_backbone:
            decay_backbone.append(param)
        elif no_decay:
            no_decay_head.append(param)
        else:
            decay_head.append(param)

    return [
        {"params": decay_backbone, "lr": backbone_lr, "weight_decay": weight_decay},
        {"params": no_decay_backbone, "lr": backbone_lr, "weight_decay": 0.0},
        {"params": decay_head, "lr": head_lr, "weight_decay": weight_decay},
        {"params": no_decay_head, "lr": head_lr, "weight_decay": 0.0},
    ]


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