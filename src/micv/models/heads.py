from __future__ import annotations

from collections.abc import Sequence

from torch import Tensor, nn


class ProjectionHead(nn.Module):
    """Maps concatenated committee features into the per-stream latent space."""

    def __init__(self, input_dim: int, latent_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.layers(features)


class MLPClassifier(nn.Module):
    """Binary classifier head that returns logits."""

    def __init__(self, input_dim: int, hidden_dims: Sequence[int], dropout: float = 0.0) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(current_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, 1))
        self.layers = nn.Sequential(*layers)

    def forward(self, features: Tensor) -> Tensor:
        return self.layers(features).squeeze(-1)