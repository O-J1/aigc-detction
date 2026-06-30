from __future__ import annotations

import torch
from torch import nn

from micv.training.scheduler import build_param_groups


class _ParamGroupModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stream1 = nn.Module()
        self.stream1.backbones = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(3, 4, kernel_size=3, bias=True),
                    nn.LayerNorm([4, 6, 6]),
                )
            ]
        )
        self.head = nn.Sequential(nn.Linear(4, 2), nn.LayerNorm(2))
        self.frozen = nn.Parameter(torch.ones(1), requires_grad=False)


def test_build_param_groups_splits_backbone_head_and_no_decay() -> None:
    model = _ParamGroupModel()

    groups = build_param_groups(model, backbone_lr=1.0e-6, head_lr=1.0e-5, weight_decay=0.02)

    assert [group["lr"] for group in groups] == [1.0e-6, 1.0e-6, 1.0e-5, 1.0e-5]
    assert [group["weight_decay"] for group in groups] == [0.02, 0.0, 0.02, 0.0]

    parameter_names = {id(param): name for name, param in model.named_parameters()}
    grouped_names = [[parameter_names[id(param)] for param in group["params"]] for group in groups]

    assert grouped_names == [
        ["stream1.backbones.0.0.weight"],
        [
            "stream1.backbones.0.0.bias",
            "stream1.backbones.0.1.weight",
            "stream1.backbones.0.1.bias",
        ],
        ["head.0.weight"],
        ["head.0.bias", "head.1.weight", "head.1.bias"],
    ]
    assert "frozen" not in {name for names in grouped_names for name in names}