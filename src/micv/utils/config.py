from __future__ import annotations

from dataclasses import MISSING, asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from types import UnionType
from typing import Any, Mapping, Union, get_args, get_origin, get_type_hints

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
    output_mode: str = "patch_tokens"
    include_cls_token: bool = False
    include_register_tokens: bool = False


@dataclass
class StreamSettings:
    name: str = ""
    backbones: list[BackboneSettings] = field(default_factory=list)
    repeat: int | None = None
    backbone: BackboneSettings | None = None


@dataclass
class ModelSettings:
    use_dummy_backbone: bool = False
    dummy_feature_dim: int = 64
    stream1_backbones: int = 4
    stream2_backbones: int = 2
    latent_dim: int = 1024
    mlp_hidden_dims: list[int] = field(default_factory=lambda: [1024, 256])
    dropout: float = 0.2
    token_pooling: str = "attention"
    stream_fusion: str = "token_concat_attention"
    streams: list[StreamSettings] = field(default_factory=list)
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
    bad_image_policy: str = "raise"


@dataclass
class AugmentationStageSettings:
    enabled: bool = True
    clean_prob: float = 0.30
    max_ops: int = 5
    severity: str = "mixed"
    op_pool: str | None = None
    intensity: str | None = None


@dataclass
class AugmentationSettings:
    train_difficulty: str = "mixed"
    static_val_augmentation: bool = True
    clean_prob: float = 0.30
    max_ops: int = 5
    pre_crop: AugmentationStageSettings = field(
        default_factory=lambda: AugmentationStageSettings(
            clean_prob=0.60,
            max_ops=2,
            severity="train",
        )
    )
    post_crop: AugmentationStageSettings | None = None
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
    amp_dtype: str = "fp16"
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
    valid_fields = {field.name: field for field in fields(instance)}
    type_hints = get_type_hints(type(instance))
    for key, value in values.items():
        if key not in valid_fields:
            raise ValueError(f"Unknown config key for {type(instance).__name__}: {key}")
        field_info = valid_fields[key]
        annotation = type_hints.get(key, field_info.type)
        current_value = getattr(instance, key)
        if is_dataclass(current_value) and isinstance(value, Mapping):
            _update_dataclass(current_value, value)
        else:
            setattr(instance, key, _coerce_config_value(annotation, value))


def _coerce_config_value(annotation: Any, value: Any) -> Any:
    if not isinstance(value, list):
        if isinstance(value, Mapping):
            dataclass_type = _dataclass_type(annotation)
            if dataclass_type is not None:
                return _build_dataclass(dataclass_type, value)
        return value

    origin = get_origin(annotation)
    if origin is not list:
        return value

    (item_type,) = get_args(annotation) or (Any,)
    dataclass_type = _dataclass_type(item_type)
    if dataclass_type is None:
        return value

    return [
        _build_dataclass(dataclass_type, item) if isinstance(item, Mapping) else item
        for item in value
    ]


def _dataclass_type(annotation: Any) -> type[Any] | None:
    if isinstance(annotation, type) and is_dataclass(annotation):
        return annotation

    origin = get_origin(annotation)
    if origin not in {Union, UnionType}:
        return None

    for arg in get_args(annotation):
        if isinstance(arg, type) and is_dataclass(arg):
            return arg
    return None


def _build_dataclass(dataclass_type: type[Any], values: Mapping[str, Any]) -> Any:
    init_values: dict[str, Any] = {}
    for field_info in fields(dataclass_type):
        if field_info.default is not MISSING:
            init_values[field_info.name] = field_info.default
        elif field_info.default_factory is not MISSING:
            init_values[field_info.name] = field_info.default_factory()

    instance = dataclass_type(**init_values)
    _update_dataclass(instance, values)
    return instance