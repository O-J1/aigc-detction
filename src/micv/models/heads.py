# src/micv/models/heads.py

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn


class TokenProjectionHead(nn.Module):
    """
    Slot-specific fusion:
      [B,N,C] * k -> concat -> B,N,sum(C_i) -> Linear -> B,N,D
    """

    def __init__(
        self,
        slot_dims: Sequence[int],
        latent_dim: int,
        dropout: float = 0.0,
        normalize_slots: bool = True,
    ) -> None:
        super().__init__()
        self.slot_dims = list(slot_dims)
        self.slot_norms = nn.ModuleList(
            [nn.LayerNorm(dim) if normalize_slots else nn.Identity() for dim in self.slot_dims]
        )
        self.proj = nn.Sequential(
            nn.Linear(sum(self.slot_dims), latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, slot_features: Sequence[Tensor]) -> Tensor:
        if len(slot_features) != len(self.slot_dims):
            raise ValueError(f"Expected {len(self.slot_dims)} slots, got {len(slot_features)}")

        token_counts = {features.shape[1] for features in slot_features}
        if len(token_counts) != 1:
            raise ValueError(f"All DINO slots must produce the same token count, got {token_counts}")

        normalized = [
            norm(features)
            for norm, features in zip(self.slot_norms, slot_features, strict=True)
        ]
        fused = torch.cat(normalized, dim=-1)
        return self.proj(fused)


class MeanTokenPool(nn.Module):
    def forward(self, tokens: Tensor) -> Tensor:
        return tokens.mean(dim=1)


class AttentionTokenPool(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, tokens: Tensor) -> Tensor:
        weights = torch.softmax(self.score(tokens), dim=1)
        return (tokens * weights).sum(dim=1)


class MLPClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Sequence[int], dropout: float = 0.0) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            dim = hidden_dim

        layers.append(nn.Linear(dim, 1))
        self.layers = nn.Sequential(*layers)

    def forward(self, features: Tensor) -> Tensor:
        return self.layers(features).squeeze(-1)