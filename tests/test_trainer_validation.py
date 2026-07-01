from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader

import micv.training.trainer as trainer_module
from micv.training.trainer import Trainer


class TinyValidationModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.logit = nn.Parameter(torch.tensor(1.0))

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        probabilities = torch.sigmoid(self.logit).expand(images.shape[0])
        return {"fused_prob": probabilities}


def test_validation_does_not_accumulate_metrics_on_non_main_rank(monkeypatch, tmp_path) -> None:
    gathered_values: list[torch.Tensor] = []

    class FailingMetrics:
        def update(self, probabilities, targets) -> None:
            raise AssertionError("non-main rank should not update metrics")

        def compute(self):
            raise AssertionError("non-main rank should not compute metrics")

    def gather(values: torch.Tensor) -> torch.Tensor:
        gathered_values.append(values.detach().cpu())
        return values

    monkeypatch.setattr(trainer_module, "is_main_process", lambda: False)
    monkeypatch.setattr(trainer_module, "all_gather_1d_tensor", gather)
    monkeypatch.setattr(trainer_module, "BinaryClassificationMetrics", FailingMetrics)

    trainer = _make_trainer(tmp_path)

    result = trainer.evaluate(_validation_loader())

    assert result is None
    assert len(gathered_values) == 2


def test_validation_accumulates_metrics_on_main_rank(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(trainer_module, "is_main_process", lambda: True)
    monkeypatch.setattr(trainer_module, "all_gather_1d_tensor", lambda values: values)

    trainer = _make_trainer(tmp_path)

    result = trainer.evaluate(_validation_loader())

    assert result is not None
    assert result.accuracy == 1.0
    assert result.tp == 1


def _make_trainer(tmp_path) -> Trainer:
    model = TinyValidationModel()
    data_loader = _validation_loader()
    return Trainer(
        model=model,
        train_loader=data_loader,
        val_loader=data_loader,
        loss_fn=nn.MSELoss(),
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        scheduler=None,
        device=torch.device("cpu"),
        output_dir=tmp_path,
        amp=False,
    )


def _validation_loader() -> DataLoader:
    batch = {
        "image": torch.ones(1, 3, 4, 4),
        "label": torch.tensor([1.0]),
    }
    return DataLoader([batch], batch_size=None)