from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader

import micv.training.trainer as trainer_module
from micv.training.losses import CombinedMICVLoss
from micv.training.trainer import Trainer, load_checkpoint


class TinyValidationModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.logit = nn.Parameter(torch.tensor(1.0))

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        probabilities = torch.sigmoid(self.logit).expand(images.shape[0])
        return {"fused_prob": probabilities}


class SignalModel(nn.Module):
    """Predicts from the mean pixel value so per-sample outputs differ."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        logits = self.scale * images.mean(dim=(1, 2, 3))
        return {"fused_prob": torch.sigmoid(logits)}


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


def _signal_loader(labels: list[float]) -> DataLoader:
    images = torch.stack(
        [
            torch.full((3, 4, 4), 2.0),
            torch.full((3, 4, 4), -2.0),
        ]
    )
    batch = {"image": images, "label": torch.tensor(labels)}
    return DataLoader([batch], batch_size=None)


def test_best_checkpoint_selected_on_clean_metric_with_degraded_reported(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(trainer_module, "is_main_process", lambda: True)
    monkeypatch.setattr(trainer_module, "all_gather_1d_tensor", lambda values: values)

    model = SignalModel()
    trainer = Trainer(
        model=model,
        train_loader=_signal_loader([1.0, 0.0]),
        val_loader=_signal_loader([1.0, 0.0]),
        degraded_val_loader=_signal_loader([0.0, 1.0]),
        loss_fn=CombinedMICVLoss(),
        optimizer=torch.optim.SGD(model.parameters(), lr=0.0),
        scheduler=None,
        device=torch.device("cpu"),
        output_dir=tmp_path,
        amp=False,
    )

    trainer.fit(epochs=1)

    assert trainer.best_roc_auc == 1.0
    checkpoint = torch.load(tmp_path / "best.pt")
    assert checkpoint["metrics"]["roc_auc"] == 1.0
    assert checkpoint["degraded_metrics"]["roc_auc"] == 0.0


def test_swa_finalize_saves_loadable_checkpoint_with_metrics(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(trainer_module, "is_main_process", lambda: True)
    monkeypatch.setattr(trainer_module, "all_gather_1d_tensor", lambda values: values)

    model = SignalModel()
    trainer = Trainer(
        model=model,
        train_loader=_signal_loader([1.0, 0.0]),
        val_loader=_signal_loader([1.0, 0.0]),
        loss_fn=CombinedMICVLoss(),
        optimizer=torch.optim.SGD(model.parameters(), lr=0.0),
        scheduler=None,
        device=torch.device("cpu"),
        output_dir=tmp_path,
        amp=False,
        swa_enabled=True,
        swa_start_epoch=1,
    )

    trainer.fit(epochs=1)

    swa_path = tmp_path / "swa_model.pt"
    assert swa_path.exists()
    checkpoint = torch.load(swa_path)
    assert checkpoint["metrics"] is not None
    assert checkpoint["metrics"]["roc_auc"] == 1.0

    target = SignalModel()
    assert load_checkpoint(swa_path, model=target) == 1
    assert torch.equal(target.scale, model.scale)


def test_load_checkpoint_accepts_wrapped_and_raw_state_dicts(tmp_path) -> None:
    source = nn.Linear(4, 1)

    raw_path = tmp_path / "raw.pt"
    torch.save(source.state_dict(), raw_path)
    raw_target = nn.Linear(4, 1)
    assert load_checkpoint(raw_path, model=raw_target) == 0
    assert torch.equal(raw_target.weight, source.weight)

    wrapped_path = tmp_path / "wrapped.pt"
    torch.save({"epoch": 5, "model": source.state_dict()}, wrapped_path)
    wrapped_target = nn.Linear(4, 1)
    assert load_checkpoint(wrapped_path, model=wrapped_target) == 5
    assert torch.equal(wrapped_target.weight, source.weight)