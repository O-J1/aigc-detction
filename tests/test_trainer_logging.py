from __future__ import annotations

import pytest
import torch

from micv.training.metrics import BinaryMetricResult
from micv.training.trainer import _format_epoch_summary, _resolve_amp_dtype


def test_resolve_amp_dtype_accepts_supported_names() -> None:
    assert _resolve_amp_dtype("bf16") is torch.bfloat16
    assert _resolve_amp_dtype("fp16") is torch.float16


def test_resolve_amp_dtype_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="Unsupported amp dtype: fp32"):
        _resolve_amp_dtype("fp32")


def test_format_epoch_summary_includes_validation_metrics() -> None:
    metric_result = BinaryMetricResult(
        roc_auc=0.81234,
        accuracy=0.75,
        precision=0.8,
        recall=0.66667,
        f1=0.72727,
        threshold=0.5,
        tp=8,
        tn=7,
        fp=2,
        fn=4,
    )

    summary = _format_epoch_summary(
        epoch_index=1,
        epochs=5,
        train_loss=0.45678,
        metric_result=metric_result,
    )

    assert "epoch 2/5 train_loss=0.4568" in summary
    assert "val_roc_auc=0.8123" in summary
    assert "val_accuracy=0.7500" in summary
    assert "val_precision=0.8000" in summary
    assert "val_recall=0.6667" in summary
    assert "val_f1=0.7273" in summary
    assert "tp=8 tn=7 fp=2 fn=4" in summary


def test_format_epoch_summary_reports_skipped_validation() -> None:
    summary = _format_epoch_summary(
        epoch_index=0,
        epochs=3,
        train_loss=0.12345,
        metric_result=None,
    )

    assert summary == "epoch 1/3 train_loss=0.1235 val=skipped"
