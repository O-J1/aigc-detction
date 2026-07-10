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
        self.probabilities.append(
            probabilities.detach().flatten().to(device="cpu", dtype=torch.float32)
        )
        self.targets.append(targets.detach().flatten().to(device="cpu", dtype=torch.float32))

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
        return _rank_roc_auc(targets, probabilities)
    return float(roc_auc_score(targets, probabilities))


def _rank_roc_auc(targets: np.ndarray, probabilities: np.ndarray) -> float:
    """Mann-Whitney U based ROC AUC with tie-averaged ranks (sklearn-free)."""
    ranks = _average_ranks(probabilities.astype(np.float64))
    positive_mask = targets == 1
    num_positive = int(positive_mask.sum())
    num_negative = int(targets.size - num_positive)
    positive_rank_sum = float(ranks[positive_mask].sum())
    u_statistic = positive_rank_sum - num_positive * (num_positive + 1) / 2.0
    return u_statistic / (num_positive * num_negative)


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    index = 0
    while index < values.size:
        tie_end = index
        while tie_end + 1 < values.size and sorted_values[tie_end + 1] == sorted_values[index]:
            tie_end += 1
        ranks[order[index : tie_end + 1]] = 0.5 * (index + tie_end) + 1.0
        index = tie_end + 1
    return ranks