from __future__ import annotations

from contextlib import nullcontext
import random
import warnings
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as distributed
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm

from micv.training.distributed import (
    all_gather_1d_tensor,
    get_rank,
    is_distributed,
    is_main_process,
)
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
        degraded_val_loader: DataLoader | None = None,
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
        self.degraded_val_loader = degraded_val_loader
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
        self.scaler = (
            _make_grad_scaler(device) if self.amp and self.amp_dtype == torch.float16 else None
        )

        self.swa_enabled = swa_enabled
        self.swa_start_epoch = swa_start_epoch
        # Keep the averaged copy on CPU so a multi-backbone ensemble does not
        # double its GPU memory footprint.
        self.swa_model = (
            AveragedModel(_unwrap_model(model), device=torch.device("cpu")) if swa_enabled else None
        )
        # anneal_epochs=1 so the first SWA-phase step lands directly on swa_lr;
        # the default of 10 would never reach it within 2-3 SWA epochs.
        self.swa_scheduler = (
            SWALR(optimizer, swa_lr=swa_learning_rate, anneal_epochs=1) if swa_enabled else None
        )

        if is_main_process():
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_checkpoint(
        self,
        checkpoint_path: str | Path,
        map_location: str | torch.device = "cpu",
        restore_rng_state: bool = True,
    ) -> int:
        return load_checkpoint(
            checkpoint_path,
            trainer=self,
            map_location=map_location,
            restore_rng_state=restore_rng_state,
        )

    def fit(self, epochs: int, start_epoch: int = 0) -> None:
        for epoch_index in range(start_epoch, epochs):
            sampler = getattr(self.train_loader, "sampler", None)
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch_index)

            # Switch to the SWA learning rate before the epoch trains so every
            # SWA epoch actually runs at swa_lr.
            if self._swa_active(epoch_index) and self.swa_scheduler is not None:
                with warnings.catch_warnings():
                    # Stepping ahead of the epoch is deliberate; silence the
                    # scheduler-before-optimizer ordering warning it triggers
                    # when SWA starts on the very first epoch.
                    warnings.filterwarnings(
                        "ignore",
                        message=r"Detected call of `lr_scheduler.step\(\)` before `optimizer.step\(\)`",
                        category=UserWarning,
                    )
                    self.swa_scheduler.step()

            train_loss = self.train_one_epoch(epoch_index)
            should_validate = self.val_loader is not None and (
                (epoch_index + 1) % self.validate_every_epochs == 0 or (epoch_index + 1) == epochs
            )
            metric_result = self.evaluate(self.val_loader) if should_validate else None
            degraded_metric_result = (
                self.evaluate(self.degraded_val_loader)
                if should_validate and self.degraded_val_loader is not None
                else None
            )

            if self._swa_active(epoch_index) and self.swa_model is not None:
                self.swa_model.update_parameters(_unwrap_model(self.model))

            rng_states = _gather_rng_states()
            if is_main_process():
                tqdm.write(
                    _format_epoch_summary(
                        epoch_index, epochs, train_loss, metric_result, degraded_metric_result
                    )
                )
                is_new_best = (
                    metric_result is not None and metric_result.roc_auc > self.best_roc_auc
                )
                if is_new_best:
                    self.best_roc_auc = metric_result.roc_auc
                self._save_checkpoint(
                    epoch_index,
                    train_loss,
                    metric_result,
                    degraded_metric_result,
                    rng_states=rng_states,
                    name="latest.pt",
                )
                if is_new_best:
                    self._save_checkpoint(
                        epoch_index,
                        train_loss,
                        metric_result,
                        degraded_metric_result,
                        rng_states=rng_states,
                        name="best.pt",
                    )

        self._finalize_swa(epochs)

    def _finalize_swa(self, epochs: int) -> None:
        if not self.swa_enabled or self.swa_model is None:
            return
        swa_module = self.swa_model.module.to(self.device)
        if self.train_loader is not None:
            _update_batch_norm(self.train_loader, swa_module, self.device)
        swa_metric = (
            self.evaluate(self.val_loader, model=swa_module)
            if self.val_loader is not None
            else None
        )
        swa_degraded_metric = (
            self.evaluate(self.degraded_val_loader, model=swa_module)
            if self.degraded_val_loader is not None
            else None
        )
        if is_main_process():
            if swa_metric is not None:
                tqdm.write(
                    "swa "
                    f"val_roc_auc={swa_metric.roc_auc:.4f} "
                    f"val_accuracy={swa_metric.accuracy:.4f}"
                    + (
                        f" degraded_val_roc_auc={swa_degraded_metric.roc_auc:.4f}"
                        if swa_degraded_metric is not None
                        else ""
                    )
                )
            checkpoint = {
                "checkpoint_version": 2,
                "epoch": epochs,
                "model": swa_module.state_dict(),
                "optimizer": None,
                "scheduler": None,
                "scaler": None,
                "train_loss": None,
                "metrics": _metric_to_dict(swa_metric),
                "degraded_metrics": _metric_to_dict(swa_degraded_metric),
                "best_roc_auc": self.best_roc_auc,
                "backbones": _collect_backbone_metadata(swa_module),
            }
            torch.save(checkpoint, self.output_dir / "swa_model.pt")

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
            should_step = (
                step_index % self.gradient_accumulation_steps == 0 or step_index == num_batches
            )
            group_start = (
                (step_index - 1) // self.gradient_accumulation_steps
            ) * self.gradient_accumulation_steps + 1
            group_size = min(
                self.gradient_accumulation_steps,
                num_batches - group_start + 1,
            )
            sync_context = (
                self.model.no_sync()
                if not should_step and hasattr(self.model, "no_sync")
                else nullcontext()
            )
            with sync_context:
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=self.amp_dtype,
                    enabled=self.amp,
                ):
                    outputs = self.model(images)
                    raw_loss = self.loss_fn(outputs, targets)
                    loss = raw_loss / group_size

                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

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

            batch_loss = float(raw_loss.detach().item())
            running_loss += batch_loss
            if is_main_process() and step_index % self.log_every_steps == 0:
                progress.set_postfix(loss=f"{running_loss / step_index:.4f}")

        return running_loss / max(1, num_batches)

    @torch.no_grad()
    def evaluate(
        self, data_loader: DataLoader, model: nn.Module | None = None
    ) -> BinaryMetricResult | None:
        eval_model = model if model is not None else self.model
        eval_model.eval()
        should_compute_metrics = is_main_process()
        local_probabilities: list[Tensor] = []
        local_targets: list[Tensor] = []
        progress = tqdm(data_loader, disable=not is_main_process(), desc="validate")
        for batch in progress:
            images, targets = self._prepare_batch(batch)
            with torch.autocast(
                device_type=self.device.type,
                dtype=self.amp_dtype,
                enabled=self.amp,
            ):
                outputs = eval_model(images)
            local_probabilities.append(outputs["fused_prob"].detach().flatten().float())
            local_targets.append(targets.detach().flatten().float())

        probabilities = (
            torch.cat(local_probabilities)
            if local_probabilities
            else torch.empty(0, device=self.device)
        )
        targets = torch.cat(local_targets) if local_targets else torch.empty(0, device=self.device)
        probabilities, targets = _trim_distributed_padding(data_loader, probabilities, targets)
        gathered_probabilities = all_gather_1d_tensor(probabilities)
        gathered_targets = all_gather_1d_tensor(targets)
        if not should_compute_metrics:
            return None
        metrics = BinaryClassificationMetrics()
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
        degraded_metric_result: BinaryMetricResult | None,
        rng_states: list[dict[str, Any]],
        name: str,
    ) -> None:
        checkpoint = {
            "checkpoint_version": 2,
            "epoch": epoch_index + 1,
            "model": _unwrap_model(self.model).state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
            "scaler": self.scaler.state_dict() if self.scaler is not None else None,
            "train_loss": train_loss,
            "metrics": _metric_to_dict(metric_result),
            "degraded_metrics": _metric_to_dict(degraded_metric_result),
            "best_roc_auc": self.best_roc_auc,
            "swa_model": self.swa_model.state_dict() if self.swa_model is not None else None,
            "swa_scheduler": (
                self.swa_scheduler.state_dict() if self.swa_scheduler is not None else None
            ),
            "rng_states": rng_states,
            "backbones": _collect_backbone_metadata(self.model),
        }
        torch.save(checkpoint, self.output_dir / name)


def load_checkpoint(
    checkpoint_path: str | Path,
    model: nn.Module | None = None,
    optimizer: Optimizer | None = None,
    scheduler: LRScheduler | None = None,
    scaler: Any | None = None,
    map_location: str | torch.device = "cpu",
    trainer: Trainer | None = None,
    restore_rng_state: bool = False,
) -> int:
    if trainer is not None:
        model = trainer.model
        optimizer = trainer.optimizer
        scheduler = trainer.scheduler
        scaler = trainer.scaler
    if model is None:
        raise ValueError("model or trainer must be provided when loading a checkpoint.")

    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=True)
    if isinstance(checkpoint, dict) and "model" in checkpoint and not _is_state_dict(checkpoint):
        state_dict = checkpoint["model"]
    else:
        # Raw state dict (e.g. externally exported weights).
        state_dict = checkpoint
        checkpoint = {}
    _unwrap_model(model).load_state_dict(state_dict)
    _warn_on_backbone_metadata_mismatch(checkpoint.get("backbones"), model)
    if optimizer is not None and checkpoint.get("optimizer") is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if scaler is not None and checkpoint.get("scaler") is not None:
        scaler.load_state_dict(checkpoint["scaler"])
    if trainer is not None:
        if trainer.swa_model is not None and checkpoint.get("swa_model") is not None:
            trainer.swa_model.load_state_dict(checkpoint["swa_model"])
        if trainer.swa_scheduler is not None and checkpoint.get("swa_scheduler") is not None:
            trainer.swa_scheduler.load_state_dict(checkpoint["swa_scheduler"])
        trainer.best_roc_auc = _checkpoint_best_roc_auc(checkpoint)
    if restore_rng_state and checkpoint.get("rng_states") is not None:
        _restore_rank_rng_state(checkpoint["rng_states"])
    return int(checkpoint.get("epoch", 0))


def _checkpoint_best_roc_auc(checkpoint: dict[str, Any]) -> float:
    value = checkpoint.get("best_roc_auc")
    if value is not None:
        return float(value)
    metrics = checkpoint.get("metrics")
    if isinstance(metrics, dict) and metrics.get("roc_auc") is not None:
        return float(metrics["roc_auc"])
    return float("-inf")


def _capture_rng_state() -> dict[str, Any]:
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": numpy_state[0],
            "state": numpy_state[1].tolist(),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _gather_rng_states() -> list[dict[str, Any]]:
    local_state = _capture_rng_state()
    if not is_distributed():
        return [local_state]
    gathered: list[dict[str, Any] | None] = [None] * distributed.get_world_size()
    distributed.all_gather_object(gathered, local_state)
    return [state for state in gathered if state is not None]


def _restore_rank_rng_state(rng_states: Any) -> None:
    if isinstance(rng_states, dict):
        state = rng_states
    elif isinstance(rng_states, list) and rng_states:
        rank = get_rank()
        state = rng_states[min(rank, len(rng_states) - 1)]
    else:
        return

    python_state = state.get("python")
    if python_state is not None:
        random.setstate(python_state)

    numpy_state = state.get("numpy")
    if isinstance(numpy_state, dict):
        np.random.set_state(
            (
                str(numpy_state["bit_generator"]),
                np.asarray(numpy_state["state"], dtype=np.uint32),
                int(numpy_state["position"]),
                int(numpy_state["has_gauss"]),
                float(numpy_state["cached_gaussian"]),
            )
        )

    torch_state = state.get("torch")
    if torch.is_tensor(torch_state):
        torch.set_rng_state(torch_state.cpu())

    cuda_states = state.get("torch_cuda")
    if torch.cuda.is_available() and isinstance(cuda_states, list):
        torch.cuda.set_rng_state_all([cuda_state.cpu() for cuda_state in cuda_states])


def _collect_backbone_metadata(model: nn.Module) -> list[dict[str, Any]]:
    metadata: list[dict[str, Any]] = []
    for module_index, module in enumerate(_unwrap_model(model).modules()):
        metadata_fn = getattr(module, "checkpoint_metadata", None)
        if callable(metadata_fn):
            values = dict(metadata_fn())
            values["module_index"] = module_index
            metadata.append(values)
    return metadata


def _warn_on_backbone_metadata_mismatch(
    saved_metadata: Any,
    model: nn.Module,
) -> None:
    if not isinstance(saved_metadata, list):
        return
    current_metadata = _collect_backbone_metadata(model)
    saved_revisions = [
        (item.get("model_name_or_path"), item.get("resolved_revision"))
        for item in saved_metadata
        if isinstance(item, dict)
    ]
    current_revisions = [
        (item.get("model_name_or_path"), item.get("resolved_revision")) for item in current_metadata
    ]
    if saved_revisions != current_revisions:
        warnings.warn(
            "Checkpoint backbone revisions differ from the currently loaded model configuration.",
            RuntimeWarning,
            stacklevel=2,
        )


def _is_state_dict(checkpoint: dict[str, Any]) -> bool:
    return all(torch.is_tensor(value) for value in checkpoint.values())


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
    degraded_metric_result: BinaryMetricResult | None = None,
) -> str:
    prefix = f"epoch {epoch_index + 1}/{epochs} train_loss={train_loss:.4f}"
    if metric_result is None:
        return f"{prefix} val=skipped"
    summary = (
        f"{prefix} "
        f"val_roc_auc={metric_result.roc_auc:.4f} "
        f"val_accuracy={metric_result.accuracy:.4f} "
        f"val_precision={metric_result.precision:.4f} "
        f"val_recall={metric_result.recall:.4f} "
        f"val_f1={metric_result.f1:.4f} "
        f"val_threshold={metric_result.threshold:.2f} "
        f"tp={metric_result.tp} tn={metric_result.tn} fp={metric_result.fp} fn={metric_result.fn}"
    )
    if degraded_metric_result is not None:
        summary += (
            f" degraded_val_roc_auc={degraded_metric_result.roc_auc:.4f}"
            f" degraded_val_accuracy={degraded_metric_result.accuracy:.4f}"
        )
    return summary


def _trim_distributed_padding(
    data_loader: DataLoader, probabilities: Tensor, targets: Tensor
) -> tuple[Tensor, Tensor]:
    """Drop samples DistributedSampler duplicated to even out rank shards.

    Padding indices sit at the tail of each rank's index sequence, so each rank
    keeps only its first ceil((N - rank) / world_size) samples.
    """
    sampler = getattr(data_loader, "sampler", None)
    if not isinstance(sampler, DistributedSampler) or getattr(sampler, "drop_last", False):
        return probabilities, targets
    dataset_size = len(sampler.dataset)
    valid = max(0, -(-(dataset_size - sampler.rank) // sampler.num_replicas))
    return probabilities[:valid], targets[:valid]


@torch.no_grad()
def _update_batch_norm(data_loader: DataLoader, model: nn.Module, device: torch.device) -> None:
    batch_norm_modules = [
        module
        for module in model.modules()
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
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
