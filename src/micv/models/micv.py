from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from micv.models.backbones import DINOv3Backbone, DINOv3BackboneConfig, TinyConvBackbone
from micv.models.heads import MLPClassifier, ProjectionHead


@dataclass(frozen=True)
class MICVForwardKeys:
    stream1_logits: str = "stream1_logits"
    stream2_logits: str = "stream2_logits"
    fused_logits: str = "fused_logits"
    stream1_prob: str = "stream1_prob"
    stream2_prob: str = "stream2_prob"
    fused_prob: str = "fused_prob"


class MICVDualStreamEnsemble(nn.Module):
    """Dual-stream DINOv3 committee detector from the MICV framework diagram."""

    output_keys = MICVForwardKeys()

    def __init__(
        self,
        backbone_factory: Callable[[], nn.Module],
        stream1_backbones: int = 4,
        stream2_backbones: int = 2,
        latent_dim: int = 512,
        mlp_hidden_dims: Sequence[int] = (512, 128),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if stream1_backbones < 1 or stream2_backbones < 1:
            raise ValueError("Both MICV streams must contain at least one backbone.")

        self.stream1_backbones = nn.ModuleList([backbone_factory() for _ in range(stream1_backbones)])
        self.stream2_backbones = nn.ModuleList([backbone_factory() for _ in range(stream2_backbones)])
        feature_dim = self._resolve_feature_dim()

        self.stream1_projection = ProjectionHead(feature_dim * stream1_backbones, latent_dim, dropout)
        self.stream2_projection = ProjectionHead(feature_dim * stream2_backbones, latent_dim, dropout)
        self.stream1_head = MLPClassifier(latent_dim, mlp_hidden_dims, dropout)
        self.stream2_head = MLPClassifier(latent_dim, mlp_hidden_dims, dropout)

    @classmethod
    def from_config(cls, model_config: Mapping[str, Any]) -> "MICVDualStreamEnsemble":
        use_dummy_backbone = bool(model_config.get("use_dummy_backbone", False))
        if use_dummy_backbone:
            dummy_feature_dim = int(model_config.get("dummy_feature_dim", 64))

            def backbone_factory() -> nn.Module:
                return TinyConvBackbone(feature_dim=dummy_feature_dim)

        else:
            backbone_values = dict(model_config.get("backbone", {}))
            if "model_name_or_path" not in backbone_values:
                raise ValueError("model.backbone.model_name_or_path is required for DINOv3 training.")
            backbone_config = DINOv3BackboneConfig(**backbone_values)

            def backbone_factory() -> nn.Module:
                return DINOv3Backbone(backbone_config)

        return cls(
            backbone_factory=backbone_factory,
            stream1_backbones=int(model_config.get("stream1_backbones", 4)),
            stream2_backbones=int(model_config.get("stream2_backbones", 2)),
            latent_dim=int(model_config.get("latent_dim", 512)),
            mlp_hidden_dims=tuple(model_config.get("mlp_hidden_dims", [512, 128])),
            dropout=float(model_config.get("dropout", 0.2)),
        )

    def forward(self, images: Tensor) -> dict[str, Tensor]:
        stream1_features = self._aggregate_stream(self.stream1_backbones, images)
        stream2_features = self._aggregate_stream(self.stream2_backbones, images)

        stream1_latent = self.stream1_projection(stream1_features)
        stream2_latent = self.stream2_projection(stream2_features)

        stream1_logits = self.stream1_head(stream1_latent)
        stream2_logits = self.stream2_head(stream2_latent)

        stream1_prob = torch.sigmoid(stream1_logits)
        stream2_prob = torch.sigmoid(stream2_logits)
        fused_prob = 0.5 * (stream1_prob + stream2_prob)
        fused_logits = torch.logit(fused_prob.clamp(min=1.0e-6, max=1.0 - 1.0e-6))

        return {
            self.output_keys.stream1_logits: stream1_logits,
            self.output_keys.stream2_logits: stream2_logits,
            self.output_keys.fused_logits: fused_logits,
            self.output_keys.stream1_prob: stream1_prob,
            self.output_keys.stream2_prob: stream2_prob,
            self.output_keys.fused_prob: fused_prob,
        }

    def _resolve_feature_dim(self) -> int:
        first_backbone = self.stream1_backbones[0]
        feature_dim = getattr(first_backbone, "feature_dim", None)
        if not isinstance(feature_dim, int) or feature_dim <= 0:
            raise ValueError("Backbones must expose a positive integer feature_dim attribute.")
        return feature_dim

    @staticmethod
    def _aggregate_stream(backbones: nn.ModuleList, images: Tensor) -> Tensor:
        stream_features = [backbone(images) for backbone in backbones]
        return torch.cat(stream_features, dim=-1)