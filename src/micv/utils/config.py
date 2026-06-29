from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


@dataclass
class BackboneSettings:
    model_name_or_path: str = "facebook/dinov3-vits16-pretrain-lvd1689m"
    revision: str | None = None
    cache_dir: str | None = None
    trust_remote_code: bool = True
    feature_dim: int | None = None
    pooling: str = "auto"
    freeze: bool = False
    gradient_checkpointing: bool = True


@dataclass
class ModelSettings:
    use_dummy_backbone: bool = False
    dummy_feature_dim: int = 64
    stream1_backbones: int = 4
    stream2_backbones: int = 2
    latent_dim: int = 512
    mlp_hidden_dims: list[int] = field(default_factory=lambda: [512, 128])
    dropout: float = 0.2
    backbone: BackboneSettings = field(default_factory=BackboneSettings)


@dataclass
class DataSettings:
    train_manifest: str | None = None
    val_manifest: str | None = None
    root_dir: str | None = None
    train_split: str = "train"
    val_split: str = "val"
    image_size: int = 512
    batch_size: int = 2
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    balanced_sampling: bool = True


@dataclass
class AugmentationSettings:
    train_difficulty: str = "mixed"
    static_val_augmentation: bool = False
    mean: list[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    std: list[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])


@dataclass
class TrainingSettings:
    epochs: int = 10
    learning_rate: float = 1.0e-5
    weight_decay: float = 0.02
    warmup_epochs: int = 1
    min_learning_rate: float = 1.0e-7
    amp: bool = True
    gradient_accumulation_steps: int = 1
    clip_grad_norm: float | None = 1.0
    validate_every_epochs: int = 1
    output_dir: str = "outputs/default"
    seed: int = 1337
    deterministic: bool = False
    log_every_steps: int = 50
    resume_from: str | None = None
    auxiliary_stream_loss_weight: float = 0.0


@dataclass
class SWASettings:
    enabled: bool = True
    start_epoch: int = 8
    learning_rate: float = 5.0e-6


@dataclass
class DistributedSettings:
    enabled: bool = False
    backend: str = "nccl"


@dataclass
class ExperimentConfig:
    experiment_name: str = "micv"
    model: ModelSettings = field(default_factory=ModelSettings)
    data: DataSettings = field(default_factory=DataSettings)
    augmentation: AugmentationSettings = field(default_factory=AugmentationSettings)
    training: TrainingSettings = field(default_factory=TrainingSettings)
    swa: SWASettings = field(default_factory=SWASettings)
    distributed: DistributedSettings = field(default_factory=DistributedSettings)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(config_path: str | Path) -> ExperimentConfig:
    with Path(config_path).open("r", encoding="utf-8") as config_file:
        raw_config = yaml.safe_load(config_file) or {}
    config = ExperimentConfig()
    _update_dataclass(config, raw_config)
    return config


def _update_dataclass(instance: Any, values: Mapping[str, Any]) -> None:
    valid_fields = {field.name for field in fields(instance)}
    for key, value in values.items():
        if key not in valid_fields:
            raise ValueError(f"Unknown config key for {type(instance).__name__}: {key}")
        current_value = getattr(instance, key)
        if is_dataclass(current_value) and isinstance(value, Mapping):
            _update_dataclass(current_value, value)
        else:
            setattr(instance, key, value)