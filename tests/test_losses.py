from __future__ import annotations

import torch

from micv.training.losses import BinaryFocalLossWithLogits


def test_binary_focal_loss_returns_scalar_and_gradients() -> None:
    logits = torch.tensor([0.0, 2.0, -2.0], requires_grad=True)
    targets = torch.tensor([0.0, 1.0, 0.0])
    loss = BinaryFocalLossWithLogits(alpha=0.5, gamma=2.0)(logits, targets)

    assert loss.ndim == 0
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()