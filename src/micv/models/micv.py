# src/micv/models/micv.py

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from micv.models.backbones import DINOv3Backbone, DINOv3BackboneConfig, TinyConvBackbone
from micv.models.heads import (
    AttentionTokenPool,
    MeanTokenPool,
    MLPClassifier,
    PooledConcatHead,
    TokenProjectionHead,
)


BackboneFactory = Callable[[], nn.Module]


@dataclass(frozen=True)
class ResolvedStreamConfig:
    name: str
    backbone_values: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class MICVForwardKeys:
    stream1_logits: str = "stream1_logits"
    stream2_logits: str = "stream2_logits"
    stream1_prob: str = "stream1_prob"
    stream2_prob: str = "stream2_prob"
    fused_prob: str = "fused_prob"


class MICVStream(nn.Module):
    def __init__(
        self,
        backbone_factories: Sequence[BackboneFactory],
        latent_dim: int,
        mlp_hidden_dims: Sequence[int],
        dropout: float,
        token_pooling: str = "attention",
        stream_fusion: str = "token_concat_attention",
    ) -> None:
        super().__init__()

        self.backbones = nn.ModuleList([backbone_factory() for backbone_factory in backbone_factories])
        slot_dims = [self._feature_dim(backbone) for backbone in self.backbones]

        self.stream_fusion = stream_fusion
        if stream_fusion == "token_concat_attention":
            self.fusion = TokenProjectionHead(
                slot_dims=slot_dims,
                latent_dim=latent_dim,
                dropout=dropout,
                normalize_slots=True,
            )
            self.pool = self._build_token_pool(token_pooling, latent_dim)
            self.head = MLPClassifier(latent_dim, mlp_hidden_dims, dropout)
        elif stream_fusion == "pooled_concat_mlp":
            self.fusion = PooledConcatHead(
                slot_dims=slot_dims,
                token_pooling=token_pooling,
                normalize_slots=True,
            )
            self.pool = nn.Identity()
            self.head = MLPClassifier(self.fusion.output_dim, mlp_hidden_dims, dropout)
        else:
            raise ValueError(f"Unsupported stream_fusion={stream_fusion!r}")

    def forward(self, images: Tensor) -> Tensor:
        slot_features = [backbone(images) for backbone in self.backbones]
        fused = self.fusion(slot_features)
        pooled = self.pool(fused)
        return self.head(pooled)

    @staticmethod
    def _build_token_pool(token_pooling: str, latent_dim: int) -> nn.Module:
        if token_pooling == "mean":
            return MeanTokenPool()
        if token_pooling == "attention":
            return AttentionTokenPool(latent_dim)
        raise ValueError(f"Unsupported token_pooling={token_pooling!r}")

    @staticmethod
    def _feature_dim(backbone: nn.Module) -> int:
        feature_dim = getattr(backbone, "feature_dim", None)
        if not isinstance(feature_dim, int) or feature_dim <= 0:
            raise ValueError("Each backbone must expose a positive integer feature_dim.")
        return feature_dim


class MICVDualStreamEnsemble(nn.Module):
    output_keys = MICVForwardKeys()

    def __init__(
        self,
        stream1_backbone_factories: Sequence[BackboneFactory],
        stream2_backbone_factories: Sequence[BackboneFactory],
        latent_dim: int = 512,
        mlp_hidden_dims: Sequence[int] = (512, 128),
        dropout: float = 0.2,
        token_pooling: str = "attention",
        stream_fusion: str = "token_concat_attention",
    ) -> None:
        super().__init__()

        self.stream1 = MICVStream(
            backbone_factories=stream1_backbone_factories,
            latent_dim=latent_dim,
            mlp_hidden_dims=mlp_hidden_dims,
            dropout=dropout,
            token_pooling=token_pooling,
            stream_fusion=stream_fusion,
        )
        self.stream2 = MICVStream(
            backbone_factories=stream2_backbone_factories,
            latent_dim=latent_dim,
            mlp_hidden_dims=mlp_hidden_dims,
            dropout=dropout,
            token_pooling=token_pooling,
            stream_fusion=stream_fusion,
        )

    @classmethod
    def from_config(cls, model_config: Mapping[str, Any]) -> "MICVDualStreamEnsemble":
        use_dummy_backbone = bool(model_config.get("use_dummy_backbone", False))
        stream_configs = _resolve_stream_configs(
            model_config,
            require_backbone_config=not use_dummy_backbone,
        )

        stream_factories = [
            _build_backbone_factories(stream_config.backbone_values, model_config, use_dummy_backbone)
            for stream_config in stream_configs
        ]

        return cls(
            stream1_backbone_factories=stream_factories[0],
            stream2_backbone_factories=stream_factories[1],
            latent_dim=int(model_config.get("latent_dim", 512)),
            mlp_hidden_dims=tuple(model_config.get("mlp_hidden_dims", [512, 128])),
            dropout=float(model_config.get("dropout", 0.2)),
            token_pooling=str(model_config.get("token_pooling", "attention")),
            stream_fusion=str(model_config.get("stream_fusion", "token_concat_attention")),
        )

    def forward(self, images: Tensor) -> dict[str, Tensor]:
        stream1_logits = self.stream1(images)
        stream2_logits = self.stream2(images)

        stream1_prob = torch.sigmoid(stream1_logits)
        stream2_prob = torch.sigmoid(stream2_logits)

        fused_prob = 0.5 * (stream1_prob + stream2_prob)

        return {
            self.output_keys.stream1_logits: stream1_logits,
            self.output_keys.stream2_logits: stream2_logits,
            self.output_keys.stream1_prob: stream1_prob,
            self.output_keys.stream2_prob: stream2_prob,
            self.output_keys.fused_prob: fused_prob,
        }


def _resolve_stream_configs(
    model_config: Mapping[str, Any],
    require_backbone_config: bool = True,
) -> tuple[ResolvedStreamConfig, ResolvedStreamConfig]:
    raw_streams = model_config.get("streams") or []
    if raw_streams:
        streams = [_resolve_explicit_stream(raw_stream) for raw_stream in raw_streams]
        stream_by_name = {stream.name: stream for stream in streams}
        if len(stream_by_name) != len(streams):
            raise ValueError("model.streams contains duplicate stream names.")
        expected_names = {"stream1", "stream2"}
        if set(stream_by_name) != expected_names:
            raise ValueError("model.streams must define exactly stream1 and stream2.")
        return stream_by_name["stream1"], stream_by_name["stream2"]

    backbone_values = dict(model_config.get("backbone", {}))
    if require_backbone_config and "model_name_or_path" not in backbone_values:
        raise ValueError("model.backbone.model_name_or_path is required.")

    return (
        ResolvedStreamConfig(
            name="stream1",
            backbone_values=tuple(
                dict(backbone_values) for _ in range(int(model_config.get("stream1_backbones", 4)))
            ),
        ),
        ResolvedStreamConfig(
            name="stream2",
            backbone_values=tuple(
                dict(backbone_values) for _ in range(int(model_config.get("stream2_backbones", 2)))
            ),
        ),
    )


def _resolve_explicit_stream(raw_stream: Mapping[str, Any]) -> ResolvedStreamConfig:
    name = str(raw_stream.get("name", ""))
    if name not in {"stream1", "stream2"}:
        raise ValueError("Each model.streams entry must be named stream1 or stream2.")

    raw_backbones = raw_stream.get("backbones") or []
    has_backbones = bool(raw_backbones)
    has_repeat = raw_stream.get("repeat") is not None or raw_stream.get("backbone") is not None

    if has_backbones == has_repeat:
        raise ValueError(
            f"model.streams[{name}] must define exactly one of backbones or repeat + backbone."
        )

    if has_backbones:
        backbone_values = tuple(dict(backbone) for backbone in raw_backbones)
    else:
        repeat = int(raw_stream.get("repeat", 0))
        if repeat <= 0:
            raise ValueError(f"model.streams[{name}].repeat must be positive.")
        repeated_backbone = raw_stream.get("backbone")
        if not isinstance(repeated_backbone, Mapping):
            raise ValueError(f"model.streams[{name}].backbone is required when repeat is set.")
        backbone_values = tuple(dict(repeated_backbone) for _ in range(repeat))

    if not backbone_values:
        raise ValueError(f"model.streams[{name}] must contain at least one backbone.")

    return ResolvedStreamConfig(name=name, backbone_values=backbone_values)


def _build_backbone_factories(
    stream_backbones: Sequence[Mapping[str, Any]],
    model_config: Mapping[str, Any],
    use_dummy_backbone: bool,
) -> tuple[BackboneFactory, ...]:
    if use_dummy_backbone:
        dummy_feature_dim = int(model_config.get("dummy_feature_dim", 64))

        return tuple(
            _tiny_backbone_factory(dummy_feature_dim=dummy_feature_dim)
            for _ in stream_backbones
        )

    return tuple(_dinov3_backbone_factory(backbone_values) for backbone_values in stream_backbones)


def _tiny_backbone_factory(dummy_feature_dim: int) -> BackboneFactory:
    def backbone_factory() -> nn.Module:
        return TinyConvBackbone(feature_dim=dummy_feature_dim)

    return backbone_factory


def _dinov3_backbone_factory(backbone_values: Mapping[str, Any]) -> BackboneFactory:
    values = dict(backbone_values)
    if "model_name_or_path" not in values:
        raise ValueError("Each model stream backbone requires model_name_or_path.")
    values.setdefault("output_mode", "patch_tokens")
    backbone_config = DINOv3BackboneConfig(**values)

    def backbone_factory() -> nn.Module:
        return DINOv3Backbone(backbone_config)

    return backbone_factory