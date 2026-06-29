from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor


@dataclass(frozen=True)
class BinaryMetricResult:
    roc_auc: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    threshold: float
    tp: int
    tn: int
    fp: int
    fn: int


class BinaryClassificationMetrics:
    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold
        self.probabilities: list[Tensor] = []
        self.targets: list[Tensor] = []

    def update(self, probabilities: Tensor, targets: Tensor) -> None:
        self.probabilities.append(probabilities.detach().flatten().cpu())
        self.targets.append(targets.detach().flatten().cpu())

    def compute(self) -> BinaryMetricResult:
        if not self.probabilities:
            raise ValueError("No metric values were accumulated.")

        probabilities = torch.cat(self.probabilities).numpy()
        targets = torch.cat(self.targets).numpy().astype(np.int64)
        predictions = (probabilities >= self.threshold).astype(np.int64)

        true_positive = int(((predictions == 1) & (targets == 1)).sum())
        true_negative = int(((predictions == 0) & (targets == 0)).sum())
        false_positive = int(((predictions == 1) & (targets == 0)).sum())
        false_negative = int(((predictions == 0) & (targets == 1)).sum())

        total = max(1, targets.size)
        accuracy = (true_positive + true_negative) / total
        precision = true_positive / max(1, true_positive + false_positive)
        recall = true_positive / max(1, true_positive + false_negative)
        f1 = 2.0 * precision * recall / max(1.0e-12, precision + recall)
        roc_auc = _safe_roc_auc(targets, probabilities)

        return BinaryMetricResult(
            roc_auc=roc_auc,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
            threshold=self.threshold,
            tp=true_positive,
            tn=true_negative,
            fp=false_positive,
            fn=false_negative,
        )

    def reset(self) -> None:
        self.probabilities.clear()
        self.targets.clear()


def _safe_roc_auc(targets: np.ndarray, probabilities: np.ndarray) -> float:
    if np.unique(targets).size < 2:
        return float("nan")
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        return float("nan")
    return float(roc_auc_score(targets, probabilities))