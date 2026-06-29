from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class DINOv3BackboneConfig:
    model_name_or_path: str
    revision: str | None = None
    cache_dir: str | None = None
    trust_remote_code: bool = True
    feature_dim: int | None = None
    pooling: str = "auto"
    freeze: bool = False
    gradient_checkpointing: bool = False


class DINOv3Backbone(nn.Module):
    """Thin Hugging Face wrapper that exposes a stable pooled feature tensor."""

    def __init__(self, config: DINOv3BackboneConfig) -> None:
        super().__init__()
        self.config = config
        self.model = self._load_hf_model(config)
        self.pooling = config.pooling
        self.feature_dim = config.feature_dim or self._feature_dim_from_config()

        if self.feature_dim is None:
            raise ValueError(
                "Could not infer DINOv3 feature_dim from the Hugging Face config. "
                "Set model.backbone.feature_dim in the YAML config."
            )

        if config.gradient_checkpointing and hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()

        if config.freeze:
            for parameter in self.model.parameters():
                parameter.requires_grad = False

    def forward(self, images: Tensor) -> Tensor:
        return self.forward_features(images)

    def forward_features(self, images: Tensor) -> Tensor:
        try:
            outputs = self.model(pixel_values=images, return_dict=True)
        except TypeError:
            outputs = self.model(images)
        return self._extract_features(outputs)

    def _load_hf_model(self, config: DINOv3BackboneConfig) -> nn.Module:
        try:
            from transformers import AutoModel
        except ImportError as error:
            raise ImportError(
                "transformers is required for DINOv3Backbone. Install the project with its "
                "default dependencies or enable model.use_dummy_backbone for smoke tests."
            ) from error

        model_kwargs: dict[str, Any] = {"trust_remote_code": config.trust_remote_code}
        if config.revision is not None:
            model_kwargs["revision"] = config.revision
        if config.cache_dir is not None:
            model_kwargs["cache_dir"] = config.cache_dir
        return AutoModel.from_pretrained(config.model_name_or_path, **model_kwargs)

    def _feature_dim_from_config(self) -> int | None:
        hf_config = getattr(self.model, "config", None)
        if hf_config is None:
            return None
        for attribute_name in (
            "hidden_size",
            "embed_dim",
            "hidden_dim",
            "projection_dim",
            "num_features",
        ):
            value = getattr(hf_config, attribute_name, None)
            if isinstance(value, int) and value > 0:
                return value
        return None

    def _extract_features(self, outputs: Any) -> Tensor:
        if isinstance(outputs, Mapping):
            pooler_output = outputs.get("pooler_output")
            if pooler_output is not None:
                return pooler_output
            last_hidden_state = outputs.get("last_hidden_state")
            if last_hidden_state is not None:
                return self._pool_sequence(last_hidden_state)

        pooler_output = getattr(outputs, "pooler_output", None)
        if pooler_output is not None:
            return pooler_output

        last_hidden_state = getattr(outputs, "last_hidden_state", None)
        if last_hidden_state is not None:
            return self._pool_sequence(last_hidden_state)

        if isinstance(outputs, (tuple, list)) and outputs:
            return self._pool_tensor(outputs[0])

        if torch.is_tensor(outputs):
            return self._pool_tensor(outputs)

        raise TypeError(f"Unsupported DINOv3 output type: {type(outputs)!r}")

    def _pool_sequence(self, sequence: Tensor) -> Tensor:
        if sequence.ndim != 3:
            return self._pool_tensor(sequence)
        if self.pooling == "mean":
            return sequence.mean(dim=1)
        return sequence[:, 0]

    @staticmethod
    def _pool_tensor(features: Tensor) -> Tensor:
        if features.ndim == 4:
            return features.mean(dim=(-2, -1))
        if features.ndim == 3:
            return features[:, 0]
        if features.ndim == 2:
            return features
        raise ValueError(f"Expected 2D, 3D, or 4D feature tensor, received shape {features.shape}")


class TinyConvBackbone(nn.Module):
    """Small backbone for trainer and shape smoke tests without HF access."""

    def __init__(self, feature_dim: int = 64) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, feature_dim),
            nn.LayerNorm(feature_dim),
        )

    def forward(self, images: Tensor) -> Tensor:
        return self.encoder(images)