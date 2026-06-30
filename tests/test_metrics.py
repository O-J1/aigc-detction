from __future__ import annotations

import torch

from micv.training.metrics import BinaryClassificationMetrics


def test_binary_metrics_accept_bfloat16_probabilities() -> None:
    metrics = BinaryClassificationMetrics()

    metrics.update(
        torch.tensor([0.25, 0.75], dtype=torch.bfloat16),
        torch.tensor([0.0, 1.0], dtype=torch.float32),
    )
    result = metrics.compute()

    assert metrics.probabilities[0].dtype is torch.float32
    assert result.accuracy == 1.0
    assert result.precision == 1.0
    assert result.recall == 1.0
    assert result.f1 == 1.0
    assert result.tp == 1
    assert result.tn == 1
    assert result.fp == 0
    assert result.fn == 0