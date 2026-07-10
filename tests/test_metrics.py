from __future__ import annotations

import numpy as np
import torch

from micv.training.metrics import BinaryClassificationMetrics, _rank_roc_auc


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


def test_rank_roc_auc_matches_expected_values() -> None:
    targets = np.array([1, 0, 1, 0])

    perfect = _rank_roc_auc(targets, np.array([0.9, 0.1, 0.8, 0.2]))
    inverted = _rank_roc_auc(targets, np.array([0.1, 0.9, 0.2, 0.8]))
    tied = _rank_roc_auc(targets, np.array([0.5, 0.5, 0.5, 0.5]))

    assert perfect == 1.0
    assert inverted == 0.0
    assert tied == 0.5


def test_compute_reports_valid_roc_auc_without_sklearn() -> None:
    metrics = BinaryClassificationMetrics()
    metrics.update(
        torch.tensor([0.9, 0.1, 0.8, 0.3]),
        torch.tensor([1.0, 0.0, 1.0, 0.0]),
    )

    result = metrics.compute()

    assert result.roc_auc == 1.0