# tests/test_committee_independence.py

from __future__ import annotations

import torch

from micv.models import MICVDualStreamEnsemble
from micv.training.losses import CombinedMICVLoss


def test_committee_slots_are_distinct_modules() -> None:
    model = MICVDualStreamEnsemble.from_config(
        {
            "use_dummy_backbone": True,
            "dummy_feature_dim": 16,
            "stream1_backbones": 4,
            "stream2_backbones": 2,
            "latent_dim": 32,
            "mlp_hidden_dims": [16],
            "dropout": 0.0,
            "token_pooling": "mean",
        }
    )

    slots = list(model.stream1.backbones) + list(model.stream2.backbones)
    assert len({id(slot) for slot in slots}) == 6

    first_params = [next(slot.parameters()) for slot in slots]
    assert len({param.data_ptr() for param in first_params}) == 6


def test_committee_slots_receive_nonidentical_gradients() -> None:
    torch.manual_seed(123)

    model = MICVDualStreamEnsemble.from_config(
        {
            "use_dummy_backbone": True,
            "dummy_feature_dim": 16,
            "stream1_backbones": 4,
            "stream2_backbones": 2,
            "latent_dim": 32,
            "mlp_hidden_dims": [16],
            "dropout": 0.0,
            "token_pooling": "mean",
        }
    )

    images = torch.randn(4, 3, 64, 64)
    targets = torch.tensor([0.0, 1.0, 0.0, 1.0])

    outputs = model(images)
    loss = CombinedMICVLoss()(outputs, targets)
    loss.backward()

    grads = []
    for slot in model.stream1.backbones:
        first_param = next(slot.parameters())
        grads.append(first_param.grad.detach().clone())

    # At least one pair should differ.
    assert any(not torch.allclose(grads[0], grad) for grad in grads[1:])