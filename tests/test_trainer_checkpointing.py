from __future__ import annotations

import copy
import random
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

import micv.training.trainer as trainer_module
from micv.training.trainer import Trainer, _capture_rng_state


class _OutputMSELoss(nn.Module):
    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        targets: torch.Tensor,
    ) -> torch.Tensor:
        return nn.functional.mse_loss(outputs["fused_prob"], targets)


class _RegressionModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=0.25)
        self.linear = nn.Linear(1, 1)

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        values = images.flatten(1)
        predictions = self.linear(self.dropout(values)).squeeze(-1)
        return {"fused_prob": predictions}


class _CheckpointModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"fused_prob": self.weight.expand(images.shape[0])}

    def checkpoint_metadata(self) -> dict[str, Any]:
        return {
            "model_name_or_path": "test/backbone",
            "requested_revision": "a" * 40,
            "resolved_revision": "a" * 40,
            "trust_remote_code": False,
        }


def _silent_tqdm(iterable=None, *args, **kwargs):
    del args, kwargs
    return iterable


_silent_tqdm.write = lambda *args, **kwargs: None


def _training_loader() -> DataLoader:
    samples = [
        {
            "image": torch.tensor([[[value]]], dtype=torch.float32),
            "label": torch.tensor(target, dtype=torch.float32),
        }
        for value, target in ((-2.0, -0.5), (-1.0, 0.0), (1.0, 0.75), (2.0, 1.25))
    ]
    return DataLoader(samples, batch_size=1, shuffle=True)


def _build_regression_trainer(
    output_dir: Path,
    initial_state: dict[str, torch.Tensor],
) -> Trainer:
    model = _RegressionModel()
    model.load_state_dict(copy.deepcopy(initial_state))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)
    return Trainer(
        model=model,
        train_loader=_training_loader(),
        val_loader=None,
        loss_fn=_OutputMSELoss(),
        optimizer=optimizer,
        scheduler=scheduler,
        device=torch.device("cpu"),
        output_dir=output_dir,
        amp=False,
        gradient_accumulation_steps=2,
        swa_enabled=True,
        swa_start_epoch=2,
        swa_learning_rate=0.01,
    )


def _build_checkpoint_trainer(output_dir: Path) -> Trainer:
    model = _CheckpointModel()
    loader = DataLoader(
        [{"image": torch.zeros(1, 1, 1, 1), "label": torch.zeros(1)}],
        batch_size=None,
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    return Trainer(
        model=model,
        train_loader=loader,
        val_loader=None,
        loss_fn=_OutputMSELoss(),
        optimizer=optimizer,
        scheduler=scheduler,
        device=torch.device("cpu"),
        output_dir=output_dir,
        amp=False,
        swa_enabled=True,
        swa_start_epoch=1,
        swa_learning_rate=0.01,
    )


def _reset_rng(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _assert_nested_equal(left: Any, right: Any) -> None:
    if torch.is_tensor(left):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for left_value, right_value in zip(left, right, strict=True):
            _assert_nested_equal(left_value, right_value)
    elif isinstance(left, float):
        assert left == pytest.approx(right)
    else:
        assert left == right


def test_checkpoint_restores_best_swa_schedulers_optimizer_and_rng(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(trainer_module, "is_main_process", lambda: True)
    trainer = _build_checkpoint_trainer(tmp_path / "source")
    trainer.best_roc_auc = 0.913

    trainer.model.weight.grad = torch.tensor(2.0)
    trainer.optimizer.step()
    trainer.optimizer.zero_grad(set_to_none=True)
    trainer.scheduler.step()
    assert trainer.swa_model is not None
    assert trainer.swa_scheduler is not None
    trainer.swa_model.update_parameters(trainer.model)
    trainer.swa_scheduler.step()

    _reset_rng(90210)
    trainer._save_checkpoint(
        epoch_index=2,
        train_loss=0.25,
        metric_result=None,
        degraded_metric_result=None,
        rng_states=[_capture_rng_state()],
        name="state.pt",
    )
    expected_random = random.random()
    expected_numpy = float(np.random.random())
    expected_torch = torch.rand(3)

    restored = _build_checkpoint_trainer(tmp_path / "restored")
    _reset_rng(1)
    start_epoch = restored.load_checkpoint(tmp_path / "source" / "state.pt")

    assert start_epoch == 3
    assert restored.best_roc_auc == pytest.approx(0.913)
    assert restored.swa_model is not None
    assert restored.swa_scheduler is not None
    _assert_nested_equal(restored.model.state_dict(), trainer.model.state_dict())
    _assert_nested_equal(restored.optimizer.state_dict(), trainer.optimizer.state_dict())
    _assert_nested_equal(restored.scheduler.state_dict(), trainer.scheduler.state_dict())
    _assert_nested_equal(restored.swa_model.state_dict(), trainer.swa_model.state_dict())
    _assert_nested_equal(restored.swa_scheduler.state_dict(), trainer.swa_scheduler.state_dict())
    assert random.random() == pytest.approx(expected_random)
    assert float(np.random.random()) == pytest.approx(expected_numpy)
    assert torch.equal(torch.rand(3), expected_torch)

    checkpoint = torch.load(
        tmp_path / "source" / "state.pt",
        weights_only=True,
    )
    assert checkpoint["checkpoint_version"] == 2
    assert checkpoint["backbones"][0]["resolved_revision"] == "a" * 40


def test_interrupted_training_matches_uninterrupted_training(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(trainer_module, "is_main_process", lambda: True)
    monkeypatch.setattr(trainer_module, "tqdm", _silent_tqdm)

    torch.manual_seed(77)
    initial_state = copy.deepcopy(_RegressionModel().state_dict())

    _reset_rng(1234)
    continuous = _build_regression_trainer(tmp_path / "continuous", initial_state)
    continuous.fit(epochs=3)

    _reset_rng(1234)
    interrupted = _build_regression_trainer(tmp_path / "interrupted", initial_state)
    interrupted.fit(epochs=2)

    resumed = _build_regression_trainer(tmp_path / "resumed", initial_state)
    start_epoch = resumed.load_checkpoint(tmp_path / "interrupted" / "latest.pt")
    assert start_epoch == 2
    resumed.fit(epochs=3, start_epoch=start_epoch)

    _assert_nested_equal(resumed.model.state_dict(), continuous.model.state_dict())
    _assert_nested_equal(resumed.optimizer.state_dict(), continuous.optimizer.state_dict())
    _assert_nested_equal(resumed.scheduler.state_dict(), continuous.scheduler.state_dict())
    assert resumed.swa_model is not None
    assert continuous.swa_model is not None
    _assert_nested_equal(resumed.swa_model.state_dict(), continuous.swa_model.state_dict())
    assert resumed.swa_scheduler is not None
    assert continuous.swa_scheduler is not None
    _assert_nested_equal(
        resumed.swa_scheduler.state_dict(),
        continuous.swa_scheduler.state_dict(),
    )
