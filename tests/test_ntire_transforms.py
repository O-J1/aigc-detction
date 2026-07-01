from __future__ import annotations

import random
from dataclasses import dataclass, field

import pytest
import torch
from PIL import Image
from torchvision import transforms

import micv.data.transforms as ntire_transforms
from micv.data.transforms import (
    AUG_GROUPS,
    NTIREAugmentationPolicy,
    StaticNTIREValidationPolicy,
    build_eval_transform,
    build_train_transform,
)


@dataclass
class DummyRecord:
    path: str
    metadata: dict[str, str] = field(default_factory=dict)


def test_ntire_policy_uses_expected_train_and_val_operation_sets() -> None:
    train_policy = NTIREAugmentationPolicy(severity="train")
    val_policy = NTIREAugmentationPolicy(severity="val")
    hard_policy = NTIREAugmentationPolicy(severity="hard")

    assert train_policy._ops_for_group("compression") == [
        "jpeg",
        "webp",
        "avif",
        "resize_then_codec",
    ]
    assert "pixelation" not in train_policy._ops_for_group("geometry")
    assert "pixelation" in val_policy._ops_for_group("geometry")
    assert "subtle_watermark" in val_policy._ops_for_group("capture")
    assert "print_scan" not in val_policy._ops_for_group("capture")
    assert "print_scan" in hard_policy._ops_for_group("capture")
    assert "watermark" not in AUG_GROUPS
    assert all("proxy" not in op for ops in AUG_GROUPS.values() for op in ops)


@pytest.mark.parametrize(
    "op_name",
    sorted({op for ops in AUG_GROUPS.values() for op in ops}),
)
def test_every_ntire_op_returns_rgb_and_preserves_size(op_name: str) -> None:
    image = Image.new("RGB", (32, 24), color=(128, 64, 32))

    random.seed(123)
    result = ntire_transforms.apply_op(op_name, image, severity="hard")

    assert result.mode == "RGB"
    assert result.size == image.size


def test_build_train_transform_uses_pre_and_post_resize_ntire_policies() -> None:
    transform = build_train_transform(image_size=32, difficulty="hard", clean_prob=0.25, max_ops=3)

    assert isinstance(transform.transforms[0], NTIREAugmentationPolicy)
    assert transform.transforms[0].severity == "train"
    assert isinstance(transform.transforms[1], transforms.RandomResizedCrop)
    assert isinstance(transform.transforms[2], NTIREAugmentationPolicy)
    assert transform.transforms[2].severity == "hard"
    assert transform.transforms[2].clean_prob == 0.25
    assert transform.transforms[2].max_ops == 3
    assert not any(
        step.__class__.__name__ == "RandomGaussianNoise" for step in transform.transforms
    )


def test_static_validation_transform_is_md5_seeded_and_restores_random_state() -> None:
    image = Image.new("RGB", (24, 24), color=(128, 64, 32))
    first_record = DummyRecord(path="/machine-a/example/path.png", metadata={"md5": "abc123"})
    second_record = DummyRecord(path="/machine-b/moved/path.png", metadata={"md5": "abc123"})
    transform = build_eval_transform(image_size=16, static_augmentation=True)

    random.seed(123)
    expected_next_random = random.random()
    random.seed(123)
    first = transform(image, first_record)
    next_random = random.random()
    random.seed(999)
    second = transform(image, second_record)

    assert getattr(transform, "needs_record", False)
    assert torch.equal(first, second)
    assert next_random == expected_next_random


def test_static_validation_policy_requests_record() -> None:
    assert StaticNTIREValidationPolicy.needs_record is True


@pytest.mark.parametrize(
    "direction",
    ["horizontal", "vertical", "diag_down", "diag_up"],
)
def test_motion_blur_supports_hard_maximum_for_every_direction(monkeypatch, direction) -> None:
    image = Image.new("RGB", (24, 24), color=(128, 64, 32))

    monkeypatch.setattr(ntire_transforms, "_randint", lambda *args, **kwargs: 15)
    monkeypatch.setattr(ntire_transforms.random, "choice", lambda choices: direction)

    result = ntire_transforms._motion_blur(image, severity="hard")

    assert result.mode == "RGB"
    assert result.size == image.size


def test_static_validation_transform_handles_hard_motion_blur_and_restores_random_state(
    monkeypatch,
) -> None:
    image = Image.new("RGB", (24, 24), color=(128, 64, 32))
    record = DummyRecord(path="example/path.png")
    transform = build_eval_transform(image_size=16, static_augmentation=True)

    def choose_motion_blur_or_direction(choices):
        if choices == ["motion_blur"]:
            return "motion_blur"
        if choices == ["horizontal", "vertical", "diag_down", "diag_up"]:
            return "diag_up"
        return choices[0]

    monkeypatch.setattr(ntire_transforms, "AUG_GROUPS", {"blur": ["motion_blur"]})
    monkeypatch.setattr(ntire_transforms, "_randint", lambda *args, **kwargs: 15)
    monkeypatch.setattr(ntire_transforms.random, "choice", choose_motion_blur_or_direction)

    random.seed(123)
    expected_next_random = random.random()
    random.seed(123)
    result = transform(image, record)
    next_random = random.random()

    assert isinstance(result, torch.Tensor)
    assert tuple(result.shape) == (3, 16, 16)
    assert next_random == expected_next_random