from __future__ import annotations

import torch

from micv.training.losses import BinaryFocalLossWithLogits, BinaryFocalLossWithProbabilities, CombinedMICVLoss


def test_binary_focal_loss_returns_scalar_and_gradients() -> None:
    logits = torch.tensor([0.0, 2.0, -2.0], requires_grad=True)
    targets = torch.tensor([0.0, 1.0, 0.0])
    loss = BinaryFocalLossWithLogits(alpha=0.5, gamma=2.0)(logits, targets)

    assert loss.ndim == 0
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_binary_focal_probability_loss_returns_scalar_and_gradients() -> None:
    probabilities = torch.tensor([0.5, 0.88, 0.12], requires_grad=True)
    targets = torch.tensor([0.0, 1.0, 0.0])
    loss = BinaryFocalLossWithProbabilities(alpha=0.5, gamma=2.0)(probabilities, targets)

    assert loss.ndim == 0
    loss.backward()
    assert probabilities.grad is not None
    assert torch.isfinite(probabilities.grad).all()


def test_combined_micv_loss_uses_fused_prob_without_fused_logits() -> None:
    fused_prob = torch.tensor([0.5, 0.88, 0.12], requires_grad=True)
    outputs = {"fused_prob": fused_prob}
    targets = torch.tensor([0.0, 1.0, 0.0])
    loss = CombinedMICVLoss()(outputs, targets)

    loss.backward()

    assert fused_prob.grad is not None
    assert torch.isfinite(fused_prob.grad).all()