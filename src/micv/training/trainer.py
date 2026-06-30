from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from micv.training.distributed import all_gather_1d_tensor, is_main_process
from micv.training.metrics import BinaryClassificationMetrics, BinaryMetricResult


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader | None,
        loss_fn: nn.Module,
        optimizer: Optimizer,
        scheduler: LRScheduler | None,
        device: torch.device,
        output_dir: str | Path,
        amp: bool = True,
        amp_dtype: str = "fp16",
        gradient_accumulation_steps: int = 1,
        clip_grad_norm: float | None = None,
        log_every_steps: int = 50,
        validate_every_epochs: int = 1,
        swa_enabled: bool = False,
        swa_start_epoch: int = 8,
        swa_learning_rate: float = 5.0e-6,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.output_dir = Path(output_dir)
        self.amp = amp and device.type == "cuda"
        self.amp_dtype = _resolve_amp_dtype(amp_dtype)
        self.gradient_accumulation_steps = max(1, gradient_accumulation_steps)
        self.clip_grad_norm = clip_grad_norm
        self.log_every_steps = log_every_steps
        self.validate_every_epochs = max(1, validate_every_epochs)
        self.best_roc_auc = float("-inf")
        self.scaler = _make_grad_scaler(device) if self.amp and self.amp_dtype == torch.float16 else None

        self.swa_enabled = swa_enabled
        self.swa_start_epoch = swa_start_epoch
        self.swa_model = AveragedModel(_unwrap_model(model)) if swa_enabled else None
        self.swa_scheduler = SWALR(optimizer, swa_lr=swa_learning_rate) if swa_enabled else None

        if is_main_process():
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def fit(self, epochs: int, start_epoch: int = 0) -> None:
        for epoch_index in range(start_epoch, epochs):
            sampler = getattr(self.train_loader, "sampler", None)
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch_index)

            train_loss = self.train_one_epoch(epoch_index)
            should_validate = (
                self.val_loader is not None
                and (
                    (epoch_index + 1) % self.validate_every_epochs == 0
                    or (epoch_index + 1) == epochs
                )
            )
            metric_result = self.evaluate(self.val_loader) if should_validate else None

            if self.swa_enabled and self.swa_model is not None and epoch_index + 1 >= self.swa_start_epoch:
                self.swa_model.update_parameters(_unwrap_model(self.model))
                if self.swa_scheduler is not None:
                    self.swa_scheduler.step()

            if is_main_process():
                tqdm.write(_format_epoch_summary(epoch_index, epochs, train_loss, metric_result))
                self._save_checkpoint(epoch_index, train_loss, metric_result, name="latest.pt")
                if metric_result is not None and metric_result.roc_auc > self.best_roc_auc:
                    self.best_roc_auc = metric_result.roc_auc
                    self._save_checkpoint(epoch_index, train_loss, metric_result, name="best.pt")

        if self.swa_enabled and self.swa_model is not None and is_main_process():
            if self.train_loader is not None:
                _update_batch_norm(self.train_loader, self.swa_model, self.device)
            torch.save(self.swa_model.module.state_dict(), self.output_dir / "swa_model.pt")

    def train_one_epoch(self, epoch_index: int) -> float:
        self.model.train()
        running_loss = 0.0
        num_batches = len(self.train_loader)
        progress = tqdm(
            enumerate(self.train_loader, start=1),
            total=num_batches,
            disable=not is_main_process(),
            desc=f"train epoch {epoch_index + 1}",
        )
        self.optimizer.zero_grad(set_to_none=True)

        for step_index, batch in progress:
            images, targets = self._prepare_batch(batch)
            with torch.autocast(
                device_type=self.device.type,
                dtype=self.amp_dtype,
                enabled=self.amp,
            ):
                outputs = self.model(images)
                loss = self.loss_fn(outputs, targets) / self.gradient_accumulation_steps

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            should_step = step_index % self.gradient_accumulation_steps == 0 or step_index == num_batches
            if should_step:
                if self.clip_grad_norm is not None:
                    if self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                if self.scheduler is not None and not self._swa_active(epoch_index):
                    self.scheduler.step()

            batch_loss = float(loss.detach().item()) * self.gradient_accumulation_steps
            running_loss += batch_loss
            if is_main_process() and step_index % self.log_every_steps == 0:
                progress.set_postfix(loss=f"{running_loss / step_index:.4f}")

        return running_loss / max(1, num_batches)

    @torch.no_grad()
    def evaluate(self, data_loader: DataLoader) -> BinaryMetricResult:
        self.model.eval()
        metrics = BinaryClassificationMetrics()
        progress = tqdm(data_loader, disable=not is_main_process(), desc="validate")
        for batch in progress:
            images, targets = self._prepare_batch(batch)
            with torch.autocast(
                device_type=self.device.type,
                dtype=self.amp_dtype,
                enabled=self.amp,
            ):
                outputs = self.model(images)
            probabilities = outputs["fused_prob"].detach().flatten()
            gathered_probabilities = all_gather_1d_tensor(probabilities)
            gathered_targets = all_gather_1d_tensor(targets.detach().flatten())
            metrics.update(gathered_probabilities.cpu(), gathered_targets.cpu())
        return metrics.compute()

    def _prepare_batch(self, batch: dict[str, Any]) -> tuple[Tensor, Tensor]:
        images = batch["image"].to(self.device, non_blocking=True)
        targets = batch["label"].to(self.device, non_blocking=True).float()
        return images, targets

    def _swa_active(self, epoch_index: int) -> bool:
        return self.swa_enabled and epoch_index + 1 >= self.swa_start_epoch

    def _save_checkpoint(
        self,
        epoch_index: int,
        train_loss: float,
        metric_result: BinaryMetricResult | None,
        name: str,
    ) -> None:
        checkpoint = {
            "epoch": epoch_index + 1,
            "model": _unwrap_model(self.model).state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
            "scaler": self.scaler.state_dict() if self.scaler is not None else None,
            "train_loss": train_loss,
            "metrics": _metric_to_dict(metric_result),
        }
        torch.save(checkpoint, self.output_dir / name)


def load_checkpoint(
    checkpoint_path: str | Path,
    model: nn.Module,
    optimizer: Optimizer | None = None,
    scheduler: LRScheduler | None = None,
    scaler: Any | None = None,
    map_location: str | torch.device = "cpu",
) -> int:
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    _unwrap_model(model).load_state_dict(checkpoint["model"])
    if optimizer is not None and checkpoint.get("optimizer") is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if scaler is not None and checkpoint.get("scaler") is not None:
        scaler.load_state_dict(checkpoint["scaler"])
    return int(checkpoint.get("epoch", 0))


def _unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


def _resolve_amp_dtype(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    raise ValueError(f"Unsupported amp dtype: {name}")


def _make_grad_scaler(device: torch.device):
    try:
        return torch.amp.GradScaler(device.type)
    except TypeError:
        return torch.cuda.amp.GradScaler()


def _metric_to_dict(metric_result: BinaryMetricResult | None) -> dict[str, Any] | None:
    if metric_result is None:
        return None
    if is_dataclass(metric_result):
        return asdict(metric_result)
    return dict(metric_result)


def _format_epoch_summary(
    epoch_index: int,
    epochs: int,
    train_loss: float,
    metric_result: BinaryMetricResult | None,
) -> str:
    prefix = f"epoch {epoch_index + 1}/{epochs} train_loss={train_loss:.4f}"
    if metric_result is None:
        return f"{prefix} val=skipped"
    return (
        f"{prefix} "
        f"val_roc_auc={metric_result.roc_auc:.4f} "
        f"val_accuracy={metric_result.accuracy:.4f} "
        f"val_precision={metric_result.precision:.4f} "
        f"val_recall={metric_result.recall:.4f} "
        f"val_f1={metric_result.f1:.4f} "
        f"val_threshold={metric_result.threshold:.2f} "
        f"tp={metric_result.tp} tn={metric_result.tn} fp={metric_result.fp} fn={metric_result.fn}"
    )


@torch.no_grad()
def _update_batch_norm(data_loader: DataLoader, model: nn.Module, device: torch.device) -> None:
    batch_norm_modules = [
        module for module in model.modules() if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
    ]
    if not batch_norm_modules:
        return

    was_training = model.training
    model.train()
    momenta: dict[nn.Module, float | None] = {}
    for module in batch_norm_modules:
        module.reset_running_stats()
        momenta[module] = module.momentum
        module.momentum = None

    for batch in data_loader:
        images = batch["image"].to(device, non_blocking=True)
        model(images)

    for module in batch_norm_modules:
        module.momentum = momenta[module]
    model.train(was_training)