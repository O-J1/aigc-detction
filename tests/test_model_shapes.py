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


class _RecordingBackbone(nn.Module):
    def __init__(self, feature_dim: int = 8, token_count: int = 4) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.token_count = token_count
        self.last_input: Tensor | None = None

    def forward(self, images: Tensor) -> Tensor:
        self.last_input = images
        return images.new_ones((images.shape[0], self.token_count, self.feature_dim))


def test_micv_stream_routes_per_backbone_views() -> None:
    stream = MICVStream(
        backbone_factories=[_RecordingBackbone] * 4,
        latent_dim=16,
        mlp_hidden_dims=[8],
        dropout=0.0,
    )
    views = torch.stack(
        [torch.full((2, 3, 8, 8), float(view_index)) for view_index in range(4)],
        dim=1,
    )

    logits = stream(views)

    assert logits.shape == (2,)
    for slot_index, backbone in enumerate(stream.backbones):
        assert backbone.last_input is not None
        assert torch.all(backbone.last_input == float(slot_index))


def test_micv_stream_reuses_last_view_when_fewer_views_than_slots() -> None:
    stream = MICVStream(
        backbone_factories=[_RecordingBackbone] * 4,
        latent_dim=16,
        mlp_hidden_dims=[8],
        dropout=0.0,
    )
    views = torch.stack(
        [torch.full((2, 3, 8, 8), float(view_index)) for view_index in range(2)],
        dim=1,
    )

    stream(views)

    expected_views = [0.0, 1.0, 1.0, 1.0]
    for backbone, expected in zip(stream.backbones, expected_views, strict=True):
        assert torch.all(backbone.last_input == expected)


def test_micv_ensemble_accepts_multi_view_input() -> None:
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

    outputs = model(torch.randn(2, 4, 3, 64, 64))

    assert outputs["fused_prob"].shape == (2,)
    assert torch.all(outputs["fused_prob"] >= 0.0)
    assert torch.all(outputs["fused_prob"] <= 1.0)


def test_micv_from_config_rejects_invalid_stream_fusion() -> None:
    with pytest.raises(ValueError, match="Unsupported stream_fusion"):
        MICVDualStreamEnsemble.from_config(
            {
                "use_dummy_backbone": True,
                "stream_fusion": "unsupported",
            }
        )


def test_micv_from_config_rejects_invalid_token_pooling() -> None:
    with pytest.raises(ValueError, match="Unsupported token_pooling"):
        MICVDualStreamEnsemble.from_config(
            {
                "use_dummy_backbone": True,
                "token_pooling": "unsupported",
            }
        )


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