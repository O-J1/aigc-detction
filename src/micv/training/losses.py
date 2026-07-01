from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as torch_functional


class BinaryFocalLossWithLogits(nn.Module):
    """Numerically stable binary focal loss for logits."""

    def __init__(self, alpha: float = 0.5, gamma: float = 2.0, reduction: str = "mean") -> None:
        super().__init__()
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError("reduction must be one of: mean, sum, none")
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        targets = targets.float()
        if targets.shape != logits.shape:
            targets = targets.view_as(logits)

        cross_entropy = torch_functional.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none",
        )
        probabilities = torch.sigmoid(logits)
        target_probabilities = probabilities * targets + (1.0 - probabilities) * (1.0 - targets)
        alpha_factor = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        focal_factor = (1.0 - target_probabilities).pow(self.gamma)
        loss = alpha_factor * focal_factor * cross_entropy

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class BinaryFocalLossWithProbabilities(nn.Module):
    """Binary focal loss for probability predictions."""

    def __init__(
        self,
        alpha: float = 0.5,
        gamma: float = 2.0,
        reduction: str = "mean",
        eps: float = 1.0e-7,
    ) -> None:
        super().__init__()
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError("reduction must be one of: mean, sum, none")
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.eps = eps

    def forward(self, probabilities: Tensor, targets: Tensor) -> Tensor:
        targets = targets.float()
        if targets.shape != probabilities.shape:
            targets = targets.view_as(probabilities)

        clamped_probabilities = probabilities.clamp(self.eps, 1.0 - self.eps)
        cross_entropy = -(
            targets * torch.log(clamped_probabilities)
            + (1.0 - targets) * torch.log(1.0 - clamped_probabilities)
        )
        target_probabilities = probabilities * targets + (1.0 - probabilities) * (1.0 - targets)
        alpha_factor = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        focal_factor = (1.0 - target_probabilities).pow(self.gamma)
        loss = alpha_factor * focal_factor * cross_entropy

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class CombinedMICVLoss(nn.Module):
    """Focal loss over fused probabilities, with optional per-stream auxiliary terms."""

    def __init__(
        self,
        fused_loss: BinaryFocalLossWithProbabilities | None = None,
        stream_loss: BinaryFocalLossWithLogits | None = None,
        fused_weight: float = 1.0,
        stream_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.fused_loss = fused_loss or BinaryFocalLossWithProbabilities()
        self.stream_loss = stream_loss or BinaryFocalLossWithLogits()
        self.fused_weight = fused_weight
        self.stream_weight = stream_weight

    def forward(self, outputs: Mapping[str, Tensor] | Tensor, targets: Tensor) -> Tensor:
        if torch.is_tensor(outputs):
            return self.fused_loss(outputs, targets)

        weighted_losses: list[Tensor] = []
        weights: list[float] = []
        if self.fused_weight > 0.0:
            weighted_losses.append(self.fused_loss(outputs["fused_prob"], targets) * self.fused_weight)
            weights.append(self.fused_weight)
        if self.stream_weight > 0.0:
            for key in ("stream1_logits", "stream2_logits"):
                weighted_losses.append(self.stream_loss(outputs[key], targets) * self.stream_weight)
                weights.append(self.stream_weight)
        if not weighted_losses:
            raise ValueError("At least one MICV loss weight must be positive.")
        return sum(weighted_losses) / sum(weights)