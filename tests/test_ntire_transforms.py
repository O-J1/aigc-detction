from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np
import pytest
import torch
from PIL import Image
from torchvision import transforms

import micv.data.transforms as ntire_transforms
from micv.transforms import TransformPolicyStage
from micv.data.transforms import (
    AUG_GROUPS,
    GeneratorConditionalNyquistNotch,
    MultiViewTransform,
    NTIREAugmentationPolicy,
    RecordAwareCompose,
    StaticNTIREValidationPolicy,
    build_eval_transform,
    build_train_transform,
    nyquist_notch,
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


def test_all_policy_ops_are_registered() -> None:
    policy_ops = ntire_transforms.TRAIN_OPS | ntire_transforms.VAL_EXTRA_OPS
    grouped_ops = {op for ops in AUG_GROUPS.values() for op in ops}

    assert policy_ops <= ntire_transforms.OPERATIONS.keys()
    assert grouped_ops <= ntire_transforms.OPERATIONS.keys()


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

    assert isinstance(transform, RecordAwareCompose)
    assert isinstance(transform.transforms[0], GeneratorConditionalNyquistNotch)
    assert isinstance(transform.transforms[1], NTIREAugmentationPolicy)
    assert transform.transforms[1].severity == "train"
    assert isinstance(transform.transforms[2], transforms.RandomResizedCrop)
    assert isinstance(transform.transforms[3], NTIREAugmentationPolicy)
    assert transform.transforms[3].severity == "hard"
    assert transform.transforms[3].clean_prob == 0.25
    assert transform.transforms[3].max_ops == 3
    assert not any(
        step.__class__.__name__ == "RandomGaussianNoise" for step in transform.transforms
    )


def test_build_train_transform_accepts_configurable_pre_and_post_crop_policies() -> None:
    transform = build_train_transform(
        image_size=32,
        pre_crop=TransformPolicyStage(enabled=False),
        post_crop=TransformPolicyStage(
            clean_prob=0.10,
            max_ops=4,
            severity="mixed",
            op_pool="all",
            intensity="hard",
        ),
    )

    policies = [step for step in transform.transforms if isinstance(step, NTIREAugmentationPolicy)]

    assert isinstance(transform.transforms[0], GeneratorConditionalNyquistNotch)
    assert isinstance(transform.transforms[1], transforms.RandomResizedCrop)
    assert len(policies) == 1
    assert policies[0].clean_prob == 0.10
    assert policies[0].max_ops == 4
    assert policies[0].op_pool == "all"
    assert policies[0].intensity == "hard"


def test_ntire_policy_separates_operation_pool_from_intensity(monkeypatch) -> None:
    policy = NTIREAugmentationPolicy(op_pool="train", intensity="hard")

    monkeypatch.setattr(ntire_transforms, "AUG_GROUPS", {"geometry": ["random_crop", "pixelation"]})

    assert policy._ops_for_group("geometry", severity_override="hard") == []


def test_static_validation_transform_is_md5_seeded_and_restores_random_state() -> None:
    image = Image.new("RGB", (24, 24), color=(128, 64, 32))
    first_record = DummyRecord(path="/machine-a/example/path.png", metadata={"md5": "abc123"})
    second_record = DummyRecord(path="/machine-b/moved/path.png", metadata={"md5": "abc123"})
    transform = build_eval_transform(image_size=16, static_augmentation=True)
    assert isinstance(transform, RecordAwareCompose)

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
    assert isinstance(transform, RecordAwareCompose)

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


def test_static_validation_policy_severity_is_configurable() -> None:
    default_policy = StaticNTIREValidationPolicy()
    val_policy = StaticNTIREValidationPolicy(severity="val")

    assert default_policy.severity == "hard"
    assert default_policy.policy_version == "val_hard_v1"
    assert val_policy.severity == "val"
    assert val_policy.policy_version == "val_val_v1"

    transform = build_eval_transform(image_size=16, static_augmentation=True, static_severity="val")
    assert isinstance(transform, RecordAwareCompose)
    assert isinstance(transform.transforms[0], StaticNTIREValidationPolicy)
    assert transform.transforms[0].severity == "val"


def test_multi_view_transform_stacks_independent_views() -> None:
    base_transform = build_train_transform(image_size=16)
    multi_view = MultiViewTransform(base_transform, num_views=3)
    image = Image.new("RGB", (32, 32), color=(200, 30, 90))

    random.seed(123)
    torch.manual_seed(123)
    result = multi_view(image)

    assert isinstance(result, torch.Tensor)
    assert tuple(result.shape) == (3, 3, 16, 16)


def test_multi_view_transform_rejects_zero_views() -> None:
    with pytest.raises(ValueError):
        MultiViewTransform(lambda image: image, num_views=0)


def _checkerboard_columns(size: int = 32) -> Image.Image:
    array = np.zeros((size, size, 3), dtype=np.uint8)
    array[:, 1::2] = 255
    return Image.fromarray(array)


def test_nyquist_notch_cancels_2px_grid_pattern() -> None:
    image = _checkerboard_columns()

    result = nyquist_notch(image)

    interior = np.asarray(result, dtype=np.float32)[4:-4, 4:-4]
    assert result.size == image.size
    assert result.mode == "RGB"
    # Alternating 0/255 columns collapse to their mean (~127.5).
    assert interior.std() < 2.0
    assert abs(interior.mean() - 127.5) < 2.0


def test_nyquist_notch_preserves_smooth_content() -> None:
    gradient = np.tile(np.linspace(30, 220, 32, dtype=np.float32), (32, 1))
    image = Image.fromarray(np.stack([gradient] * 3, axis=-1).astype(np.uint8))

    result = nyquist_notch(image)

    difference = np.abs(
        np.asarray(result, dtype=np.float32) - np.asarray(image, dtype=np.float32)
    )
    assert difference[4:-4, 4:-4].max() <= 2.0


@pytest.mark.parametrize(
    "generator",
    ["Qwen", "Qwen-Image", "qwen_image", "Anima", "Krea 2", "krea-2", "KREA2"],
)
def test_generator_conditional_notch_applies_to_target_generators(generator: str) -> None:
    notch = GeneratorConditionalNyquistNotch(probability=1.0)
    image = _checkerboard_columns()
    record = DummyRecord(path="x.png", metadata={"generator": generator})

    result = notch(image, record)

    assert result is not image
    assert np.asarray(result)[4:-4, 4:-4].std() < 2.0


@pytest.mark.parametrize("generator", ["animagine", "flux", "sdxl", "", None])
def test_generator_conditional_notch_skips_other_generators(generator) -> None:
    notch = GeneratorConditionalNyquistNotch(probability=1.0)
    image = _checkerboard_columns()
    metadata = {} if generator is None else {"generator": generator}
    record = DummyRecord(path="x.png", metadata=metadata)

    assert notch(image, record) is image
    assert notch(image, None) is image


def test_generator_conditional_notch_respects_probability(monkeypatch) -> None:
    notch = GeneratorConditionalNyquistNotch()
    image = _checkerboard_columns()
    record = DummyRecord(path="x.png", metadata={"generator": "qwen-image"})

    monkeypatch.setattr(random, "random", lambda: 0.35)
    assert notch(image, record) is image

    monkeypatch.setattr(random, "random", lambda: 0.34)
    assert notch(image, record) is not image


def test_train_transform_is_record_aware_and_works_without_record() -> None:
    transform = build_train_transform(image_size=16)

    assert getattr(transform, "needs_record", False)

    random.seed(123)
    torch.manual_seed(123)
    result = transform(Image.new("RGB", (32, 32), color=(200, 30, 90)))

    assert isinstance(result, torch.Tensor)
    assert tuple(result.shape) == (3, 16, 16)