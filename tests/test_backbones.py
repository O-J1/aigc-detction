from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from micv.models.backbones import DINOv3Backbone, DINOv3BackboneConfig


class _FakeHFModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(8, 8)
        self.dropout = nn.Dropout(p=0.5)
        self.config = SimpleNamespace(
            hidden_size=8,
            num_register_tokens=2,
            _commit_hash="b" * 40,
        )

    def forward(self, pixel_values, return_dict=True):
        del return_dict
        batch_size = pixel_values.shape[0]
        hidden = torch.ones(batch_size, 7, 8, device=pixel_values.device)
        return SimpleNamespace(
            last_hidden_state=self.dropout(self.projection(hidden)),
            pooler_output=torch.ones(batch_size, 8, device=pixel_values.device),
        )


class _FakeAutoModel:
    calls: list[tuple[str, dict[str, object]]] = []

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, **kwargs):
        cls.calls.append((model_name_or_path, kwargs))
        return _FakeHFModel()


def test_dinov3_remote_code_is_disabled_by_default() -> None:
    config = DINOv3BackboneConfig(model_name_or_path="facebook/dinov3-test")

    assert config.trust_remote_code is False


def test_dinov3_uses_native_transformers_and_records_resolved_revision(
    monkeypatch,
) -> None:
    _FakeAutoModel.calls.clear()
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoModel=_FakeAutoModel),
    )
    requested_revision = "a" * 40

    backbone = DINOv3Backbone(
        DINOv3BackboneConfig(
            model_name_or_path="facebook/dinov3-test",
            revision=requested_revision,
            freeze=True,
            output_mode="patch_tokens",
        )
    )

    assert _FakeAutoModel.calls == [
        (
            "facebook/dinov3-test",
            {
                "trust_remote_code": False,
                "revision": requested_revision,
            },
        )
    ]
    assert backbone.resolved_revision == "b" * 40
    assert backbone.checkpoint_metadata() == {
        "model_name_or_path": "facebook/dinov3-test",
        "requested_revision": requested_revision,
        "resolved_revision": "b" * 40,
        "trust_remote_code": False,
    }

    # DINOv3 has one class token and two register tokens in this fixture, so
    # patch-only mode returns the remaining four tokens.
    output = backbone(torch.zeros(2, 3, 16, 16))
    assert output.shape == (2, 4, 8)
    assert all(not parameter.requires_grad for parameter in backbone.model.parameters())

    backbone.train()
    assert backbone.training is True
    assert backbone.model.training is False


def test_dinov3_warns_when_remote_revision_is_not_immutable(monkeypatch) -> None:
    _FakeAutoModel.calls.clear()
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoModel=_FakeAutoModel),
    )

    with pytest.warns(RuntimeWarning, match="not pinned"):
        DINOv3Backbone(
            DINOv3BackboneConfig(
                model_name_or_path="facebook/dinov3-test",
                revision="main",
            )
        )
