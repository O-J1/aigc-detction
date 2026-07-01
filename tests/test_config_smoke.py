from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from micv.models import MICVDualStreamEnsemble
from micv.utils import load_config


CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


@pytest.mark.parametrize("config_path", sorted(CONFIG_DIR.glob("*.yaml")))
def test_checked_in_config_builds_dummy_model_and_runs_forward(config_path: Path) -> None:
    config = load_config(config_path)
    config.model.use_dummy_backbone = True
    config.model.dummy_feature_dim = 8
    config.model.latent_dim = 16
    config.model.mlp_hidden_dims = [8]
    config.model.dropout = 0.0

    model = MICVDualStreamEnsemble.from_config(asdict(config.model))
    outputs = model(torch.randn(2, 3, 32, 32))

    assert outputs["stream1_logits"].shape == (2,)
    assert outputs["stream2_logits"].shape == (2,)
    assert outputs["fused_prob"].shape == (2,)