from __future__ import annotations

from contextlib import contextmanager

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

import micv.training.trainer as trainer_module
from micv.training.trainer import Trainer


class _ScalarModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"fused_prob": self.weight.expand(images.shape[0])}


class _MeanOutputLoss(nn.Module):
    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        targets: torch.Tensor,
    ) -> torch.Tensor:
        del targets
        return outputs["fused_prob"].mean()


class _NoSyncWrapper(nn.Module):
    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module
        self.no_sync_calls = 0

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.module(images)

    @contextmanager
    def no_sync(self):
        self.no_sync_calls += 1
        yield


def _five_batch_loader() -> DataLoader:
    batches = [
        {
            "image": torch.zeros(1, 1, 1, 1),
            "label": torch.zeros(1),
        }
        for _ in range(5)
    ]
    return DataLoader(batches, batch_size=None)


def _make_trainer(tmp_path, model: nn.Module) -> Trainer:
    return Trainer(
        model=model,
        train_loader=_five_batch_loader(),
        val_loader=None,
        loss_fn=_MeanOutputLoss(),
        optimizer=torch.optim.SGD(model.parameters(), lr=1.0),
        scheduler=None,
        device=torch.device("cpu"),
        output_dir=tmp_path,
        amp=False,
        gradient_accumulation_steps=4,
    )


def test_partial_accumulation_group_uses_its_actual_size(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(trainer_module, "is_main_process", lambda: False)
    model = _ScalarModel()
    trainer = _make_trainer(tmp_path, model)

    trainer.train_one_epoch(epoch_index=0)

    # Four microbatches average to one gradient step, and the final one-batch
    # group receives a full-strength step rather than a quarter-strength step.
    assert model.weight.item() == pytest.approx(-1.0)


def test_gradient_accumulation_skips_ddp_sync_for_intermediate_batches(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(trainer_module, "is_main_process", lambda: False)
    model = _NoSyncWrapper(_ScalarModel())
    trainer = _make_trainer(tmp_path, model)

    trainer.train_one_epoch(epoch_index=0)

    assert model.no_sync_calls == 3
