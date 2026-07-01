from __future__ import annotations

import pytest
import torch
from torch import Tensor, nn

from micv.models import MICVDualStreamEnsemble
from micv.models.micv import MICVStream


class _StaticTokenBackbone(nn.Module):
    def __init__(self, feature_dim: int, token_count: int) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.token_count = token_count

    def forward(self, images: Tensor) -> Tensor:
        return images.new_ones((images.shape[0], self.token_count, self.feature_dim))


def test_micv_dummy_backbone_forward_shapes() -> None:
    model = MICVDualStreamEnsemble.from_config(
        {
            "use_dummy_backbone": True,
            "dummy_feature_dim": 16,
            "stream1_backbones": 4,
            "stream2_backbones": 2,
            "latent_dim": 32,
            "mlp_hidden_dims": [16],
            "dropout": 0.0,
        }
    )
    outputs = model(torch.randn(2, 3, 64, 64))

    assert outputs["stream1_logits"].shape == (2,)
    assert outputs["stream2_logits"].shape == (2,)
    assert outputs["fused_prob"].shape == (2,)
    assert "fused_logits" not in outputs
    assert torch.all(outputs["fused_prob"] >= 0.0)
    assert torch.all(outputs["fused_prob"] <= 1.0)


def test_micv_dummy_backbone_accepts_explicit_stream_backbones() -> None:
    model = MICVDualStreamEnsemble.from_config(
        {
            "use_dummy_backbone": True,
            "dummy_feature_dim": 16,
            "latent_dim": 32,
            "mlp_hidden_dims": [16],
            "dropout": 0.0,
            "streams": [
                {"name": "stream1", "backbones": [{}, {}, {}, {}]},
                {"name": "stream2", "backbones": [{}, {}]},
            ],
        }
    )

    assert len(model.stream1.backbones) == 4
    assert len(model.stream2.backbones) == 2


def test_micv_dummy_backbone_accepts_repeat_stream_backbone() -> None:
    model = MICVDualStreamEnsemble.from_config(
        {
            "use_dummy_backbone": True,
            "dummy_feature_dim": 16,
            "latent_dim": 32,
            "mlp_hidden_dims": [16],
            "dropout": 0.0,
            "streams": [
                {"name": "stream1", "repeat": 4, "backbone": {}},
                {"name": "stream2", "repeat": 2, "backbone": {}},
            ],
        }
    )

    slots = list(model.stream1.backbones) + list(model.stream2.backbones)
    assert len(slots) == 6
    assert len({id(slot) for slot in slots}) == 6


def test_pooled_concat_mlp_allows_mismatched_token_counts_and_dims() -> None:
    stream = MICVStream(
        backbone_factories=(
            lambda: _StaticTokenBackbone(feature_dim=8, token_count=5),
            lambda: _StaticTokenBackbone(feature_dim=12, token_count=3),
        ),
        latent_dim=16,
        mlp_hidden_dims=[10],
        dropout=0.0,
        token_pooling="mean",
        stream_fusion="pooled_concat_mlp",
    )

    logits = stream(torch.randn(2, 3, 64, 64))

    assert logits.shape == (2,)


def test_token_concat_attention_rejects_mismatched_token_counts() -> None:
    stream = MICVStream(
        backbone_factories=(
            lambda: _StaticTokenBackbone(feature_dim=8, token_count=5),
            lambda: _StaticTokenBackbone(feature_dim=8, token_count=3),
        ),
        latent_dim=16,
        mlp_hidden_dims=[10],
        dropout=0.0,
        token_pooling="mean",
        stream_fusion="token_concat_attention",
    )

    with pytest.raises(ValueError, match="same token count"):
        stream(torch.randn(2, 3, 64, 64))