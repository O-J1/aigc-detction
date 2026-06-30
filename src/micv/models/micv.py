# src/micv/models/micv.py

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from micv.models.backbones import DINOv3Backbone, DINOv3BackboneConfig, TinyConvBackbone
from micv.models.heads import AttentionTokenPool, MeanTokenPool, MLPClassifier, TokenProjectionHead


@dataclass(frozen=True)
class MICVForwardKeys:
    stream1_logits: str = "stream1_logits"
    stream2_logits: str = "stream2_logits"
    fused_logits: str = "fused_logits"
    stream1_prob: str = "stream1_prob"
    stream2_prob: str = "stream2_prob"
    fused_prob: str = "fused_prob"


class MICVStream(nn.Module):
    def __init__(
        self,
        backbone_factory: Callable[[], nn.Module],
        num_slots: int,
        latent_dim: int,
        mlp_hidden_dims: Sequence[int],
        dropout: float,
        token_pooling: str = "attention",
    ) -> None:
        super().__init__()

        self.backbones = nn.ModuleList([backbone_factory() for _ in range(num_slots)])
        slot_dims = [self._feature_dim(backbone) for backbone in self.backbones]

        self.projection = TokenProjectionHead(
            slot_dims=slot_dims,
            latent_dim=latent_dim,
            dropout=dropout,
            normalize_slots=True,
        )

        if token_pooling == "mean":
            self.pool = MeanTokenPool()
        elif token_pooling == "attention":
            self.pool = AttentionTokenPool(latent_dim)
        else:
            raise ValueError(f"Unsupported token_pooling={token_pooling!r}")

        self.head = MLPClassifier(latent_dim, mlp_hidden_dims, dropout)

    def forward(self, images: Tensor) -> Tensor:
        slot_tokens = [backbone(images) for backbone in self.backbones]  # each B,N,C
        fused_tokens = self.projection(slot_tokens)                     # B,N,D
        pooled = self.pool(fused_tokens)                                # B,D
        return self.head(pooled)                                        # B

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
        backbone_factory: Callable[[], nn.Module],
        stream1_backbones: int = 4,
        stream2_backbones: int = 2,
        latent_dim: int = 512,
        mlp_hidden_dims: Sequence[int] = (512, 128),
        dropout: float = 0.2,
        token_pooling: str = "attention",
    ) -> None:
        super().__init__()

        self.stream1 = MICVStream(
            backbone_factory=backbone_factory,
            num_slots=stream1_backbones,
            latent_dim=latent_dim,
            mlp_hidden_dims=mlp_hidden_dims,
            dropout=dropout,
            token_pooling=token_pooling,
        )
        self.stream2 = MICVStream(
            backbone_factory=backbone_factory,
            num_slots=stream2_backbones,
            latent_dim=latent_dim,
            mlp_hidden_dims=mlp_hidden_dims,
            dropout=dropout,
            token_pooling=token_pooling,
        )

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
                raise ValueError("model.backbone.model_name_or_path is required.")

            backbone_values.setdefault("output_mode", "patch_tokens")
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
            token_pooling=str(model_config.get("token_pooling", "attention")),
        )

    def forward(self, images: Tensor) -> dict[str, Tensor]:
        stream1_logits = self.stream1(images)
        stream2_logits = self.stream2(images)

        stream1_prob = torch.sigmoid(stream1_logits)
        stream2_prob = torch.sigmoid(stream2_logits)

        fused_prob = 0.5 * (stream1_prob + stream2_prob)
        fused_logits = torch.logit(fused_prob.clamp(1.0e-6, 1.0 - 1.0e-6))

        return {
            self.output_keys.stream1_logits: stream1_logits,
            self.output_keys.stream2_logits: stream2_logits,
            self.output_keys.fused_logits: fused_logits,
            self.output_keys.stream1_prob: stream1_prob,
            self.output_keys.stream2_prob: stream2_prob,
            self.output_keys.fused_prob: fused_prob,
        }