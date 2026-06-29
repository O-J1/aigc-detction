from __future__ import annotations

import torch

from micv.models import MICVDualStreamEnsemble


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
    assert outputs["fused_logits"].shape == (2,)
    assert outputs["fused_prob"].shape == (2,)
    assert torch.all(outputs["fused_prob"] >= 0.0)
    assert torch.all(outputs["fused_prob"] <= 1.0)