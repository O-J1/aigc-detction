from __future__ import annotations

import pytest
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


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_binary_focal_probability_loss_is_finite_at_saturated_low_precision(dtype) -> None:
    # Under fp16/bf16 autocast sigmoid can emit exactly 0.0 or 1.0; the loss
    # must clamp in float32 or log(1 - p) becomes -inf.
    probabilities = torch.tensor([0.0, 1.0, 1.0, 0.0], dtype=dtype, requires_grad=True)
    targets = torch.tensor([1.0, 0.0, 1.0, 0.0])

    loss = BinaryFocalLossWithProbabilities(alpha=0.5, gamma=2.0)(probabilities, targets)

    assert torch.isfinite(loss)
    loss.backward()
    assert probabilities.grad is not None
    assert torch.isfinite(probabilities.grad.float()).all()


def test_combined_micv_loss_supervises_stream_logits_with_positive_weight() -> None:
    stream1_logits = torch.tensor([0.5, -1.0], requires_grad=True)
    stream2_logits = torch.tensor([-0.5, 1.0], requires_grad=True)
    fused_prob = 0.5 * (torch.sigmoid(stream1_logits) + torch.sigmoid(stream2_logits))
    outputs = {
        "fused_prob": fused_prob,
        "stream1_logits": stream1_logits,
        "stream2_logits": stream2_logits,
    }
    targets = torch.tensor([1.0, 0.0])

    loss = CombinedMICVLoss(fused_weight=1.0, stream_weight=0.5)(outputs, targets)
    loss.backward()

    assert stream1_logits.grad is not None
    assert stream2_logits.grad is not None
    assert torch.isfinite(stream1_logits.grad).all()
    assert torch.isfinite(stream2_logits.grad).all()
    assert not torch.allclose(stream1_logits.grad, torch.zeros_like(stream1_logits.grad))