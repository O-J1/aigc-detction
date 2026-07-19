# src/micv/models/backbones.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Mapping
import warnings

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class DINOv3BackboneConfig:
    model_name_or_path: str
    revision: str | None = None
    cache_dir: str | None = None
    trust_remote_code: bool = False
    feature_dim: int | None = None

    # New
    output_mode: str = "patch_tokens"  # "patch_tokens" or "pooled"
    include_cls_token: bool = False
    include_register_tokens: bool = False

    # Existing
    pooling: str = "cls"  # used only for output_mode="pooled"
    freeze: bool = False
    gradient_checkpointing: bool = False


class DINOv3Backbone(nn.Module):
    def __init__(self, config: DINOv3BackboneConfig) -> None:
        super().__init__()
        self.config = config
        _warn_for_unpinned_remote_model(config)
        self.model = self._load_hf_model(config)
        self.resolved_revision = _resolved_model_revision(config, self.model)
        self.feature_dim = config.feature_dim or self._feature_dim_from_config()

        if self.feature_dim is None:
            raise ValueError("Set model.backbone.feature_dim; could not infer it.")

        if config.gradient_checkpointing and hasattr(self.model, "gradient_checkpointing_enable"):
            try:
                # Reentrant checkpointing (the historical default) breaks under
                # DDP ("parameter marked ready twice"); prefer non-reentrant.
                self.model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            except TypeError:
                self.model.gradient_checkpointing_enable()

        if config.freeze:
            for p in self.model.parameters():
                p.requires_grad = False
            self.model.eval()

    def train(self, mode: bool = True) -> DINOv3Backbone:
        super().train(mode)
        if self.config.freeze:
            # A frozen backbone must also keep stochastic layers and any
            # running-statistic modules in evaluation mode.
            self.model.eval()
        return self

    def forward(self, images: Tensor) -> Tensor:
        outputs = self._forward_hf(images)

        if self.config.output_mode == "patch_tokens":
            return self._extract_patch_tokens(outputs)

        if self.config.output_mode == "pooled":
            return self._extract_pooled(outputs)

        raise ValueError(f"Unsupported output_mode={self.config.output_mode!r}")

    def _forward_hf(self, images: Tensor) -> Any:
        try:
            return self.model(pixel_values=images, return_dict=True)
        except TypeError:
            return self.model(images)

    def _extract_patch_tokens(self, outputs: Any) -> Tensor:
        last_hidden_state = _get_output(outputs, "last_hidden_state")
        if last_hidden_state is None:
            raise TypeError("DINOv3 patch-token mode requires outputs.last_hidden_state.")

        # ConvNeXt-style feature map: B,C,H,W -> B,N,C
        if last_hidden_state.ndim == 4:
            return last_hidden_state.flatten(2).transpose(1, 2).contiguous()

        if last_hidden_state.ndim != 3:
            raise ValueError(f"Expected B,N,C token tensor, got {tuple(last_hidden_state.shape)}")

        num_registers = int(
            getattr(getattr(self.model, "config", None), "num_register_tokens", 0) or 0
        )

        pieces: list[Tensor] = []

        if self.config.include_cls_token:
            pieces.append(last_hidden_state[:, :1, :])

        if self.config.include_register_tokens and num_registers > 0:
            pieces.append(last_hidden_state[:, 1 : 1 + num_registers, :])

        patch_start = 1 + num_registers
        pieces.append(last_hidden_state[:, patch_start:, :])

        return torch.cat(pieces, dim=1) if len(pieces) > 1 else pieces[0]

    def _extract_pooled(self, outputs: Any) -> Tensor:
        pooler_output = _get_output(outputs, "pooler_output")
        if pooler_output is not None:
            return pooler_output

        last_hidden_state = _get_output(outputs, "last_hidden_state")
        if last_hidden_state is None:
            raise TypeError("Could not extract pooled DINOv3 features.")

        if last_hidden_state.ndim == 4:
            return last_hidden_state.mean(dim=(-2, -1))

        if self.config.pooling == "mean":
            return last_hidden_state.mean(dim=1)

        return last_hidden_state[:, 0]

    def _feature_dim_from_config(self) -> int | None:
        hf_config = getattr(self.model, "config", None)
        if hf_config is None:
            return None

        for name in ("hidden_size", "embed_dim", "hidden_dim", "projection_dim", "num_features"):
            value = getattr(hf_config, name, None)
            if isinstance(value, int) and value > 0:
                return value

        return None

    def _load_hf_model(self, config: DINOv3BackboneConfig) -> nn.Module:
        from transformers import AutoModel

        kwargs: dict[str, Any] = {"trust_remote_code": config.trust_remote_code}
        if config.revision is not None:
            kwargs["revision"] = config.revision
        if config.cache_dir is not None:
            kwargs["cache_dir"] = config.cache_dir

        return AutoModel.from_pretrained(config.model_name_or_path, **kwargs)

    def checkpoint_metadata(self) -> dict[str, Any]:
        return {
            "model_name_or_path": self.config.model_name_or_path,
            "requested_revision": self.config.revision,
            "resolved_revision": self.resolved_revision,
            "trust_remote_code": self.config.trust_remote_code,
        }


def _get_output(outputs: Any, name: str) -> Any:
    if isinstance(outputs, Mapping):
        return outputs.get(name)
    return getattr(outputs, name, None)


def _resolved_model_revision(config: DINOv3BackboneConfig, model: nn.Module) -> str | None:
    model_config = getattr(model, "config", None)
    resolved = getattr(model_config, "_commit_hash", None)
    if isinstance(resolved, str) and resolved:
        return resolved
    return config.revision


def _warn_for_unpinned_remote_model(config: DINOv3BackboneConfig) -> None:
    model_path = Path(config.model_name_or_path).expanduser()
    if model_path.exists():
        return
    revision = config.revision
    if revision is None or re.fullmatch(r"[0-9a-fA-F]{40}", revision) is None:
        warnings.warn(
            "Remote model revision is not pinned to a 40-character commit hash; "
            "training may not be reproducible.",
            RuntimeWarning,
            stacklevel=2,
        )


class TinyConvBackbone(nn.Module):
    """Token-output smoke backbone: B,3,H,W -> B,N,C."""

    def __init__(self, feature_dim: int = 64) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(3, feature_dim // 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(feature_dim // 2),
            nn.GELU(),
            nn.Conv2d(feature_dim // 2, feature_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.GELU(),
        )

    def forward(self, images: Tensor) -> Tensor:
        features = self.encoder(images)  # B,C,H,W
        return features.flatten(2).transpose(1, 2)  # B,N,C
