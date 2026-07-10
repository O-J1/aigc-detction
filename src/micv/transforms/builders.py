from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from micv.transforms.compose import RecordAwareCompose
from micv.transforms.defaults import DEFAULT_MEAN, DEFAULT_STD
from micv.transforms.policies import (
    GeneratorConditionalNyquistNotch,
    NTIREAugmentationPolicy,
    StaticNTIREValidationPolicy,
)


@dataclass(frozen=True)
class TransformPolicyStage:
    enabled: bool = True
    clean_prob: float = 0.30
    max_ops: int = 5
    severity: str = "mixed"
    op_pool: str | None = None
    intensity: str | None = None


class MultiViewTransform:
    """Apply a stochastic transform ``num_views`` times, stacking the results.

    Produces a [num_views, C, H, W] tensor so each committee backbone can be
    fed a differently-augmented view of the same source image.
    """

    def __init__(self, transform: Any, num_views: int) -> None:
        if num_views < 1:
            raise ValueError("num_views must be >= 1")
        self.transform = transform
        self.num_views = num_views
        self.needs_record = bool(getattr(transform, "needs_record", False))

    def __call__(self, image: Any, record: Any = None) -> torch.Tensor:
        views = [
            self.transform(image, record) if self.needs_record else self.transform(image)
            for _ in range(self.num_views)
        ]
        return torch.stack(views, dim=0)


def build_train_transform(
    image_size: int = 512,
    difficulty: str = "mixed",
    clean_prob: float = 0.30,
    max_ops: int = 5,
    mean: Sequence[float] = DEFAULT_MEAN,
    std: Sequence[float] = DEFAULT_STD,
    pre_crop: TransformPolicyStage | Mapping[str, Any] | Any | None = None,
    post_crop: TransformPolicyStage | Mapping[str, Any] | Any | None = None,
) -> RecordAwareCompose:
    pre_crop_stage = _coerce_stage(
        pre_crop,
        TransformPolicyStage(clean_prob=0.60, max_ops=2, severity="train"),
    )
    post_crop_stage = _coerce_stage(
        post_crop,
        TransformPolicyStage(clean_prob=clean_prob, max_ops=max_ops, severity=difficulty),
    )

    # Applied first, at native resolution: the 2px grid artifact lives at the
    # generator's native pixel scale and would be destroyed by resampling.
    transform_steps: list[object] = [GeneratorConditionalNyquistNotch()]
    pre_crop_policy = _policy_from_stage(pre_crop_stage)
    if pre_crop_policy is not None:
        transform_steps.append(pre_crop_policy)
    transform_steps.extend(
        [
            transforms.RandomResizedCrop(
                image_size,
                scale=(0.65, 1.0),
                ratio=(0.75, 1.333),
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            ),
        ]
    )
    post_crop_policy = _policy_from_stage(post_crop_stage)
    if post_crop_policy is not None:
        transform_steps.append(post_crop_policy)
    transform_steps.extend(
        [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return RecordAwareCompose(transform_steps)


def _coerce_stage(
    stage: TransformPolicyStage | Mapping[str, Any] | Any | None,
    default: TransformPolicyStage,
) -> TransformPolicyStage:
    if stage is None:
        return default
    if isinstance(stage, TransformPolicyStage):
        return stage
    values = {
        "enabled": getattr(stage, "enabled", default.enabled),
        "clean_prob": getattr(stage, "clean_prob", default.clean_prob),
        "max_ops": getattr(stage, "max_ops", default.max_ops),
        "severity": getattr(stage, "severity", default.severity),
        "op_pool": getattr(stage, "op_pool", default.op_pool),
        "intensity": getattr(stage, "intensity", default.intensity),
    }
    if isinstance(stage, Mapping):
        values.update(stage)
    return TransformPolicyStage(**values)


def _policy_from_stage(stage: TransformPolicyStage) -> NTIREAugmentationPolicy | None:
    if not stage.enabled:
        return None
    return NTIREAugmentationPolicy(
        clean_prob=stage.clean_prob,
        max_ops=stage.max_ops,
        severity=stage.severity,
        op_pool=stage.op_pool,
        intensity=stage.intensity,
    )


def build_eval_transform(
    image_size: int = 512,
    mean: Sequence[float] = DEFAULT_MEAN,
    std: Sequence[float] = DEFAULT_STD,
    static_augmentation: bool = False,
    static_severity: str = "hard",
) -> transforms.Compose | RecordAwareCompose:
    transform_steps: list[object] = []
    if static_augmentation:
        transform_steps.append(StaticNTIREValidationPolicy(severity=static_severity))
    transform_steps.extend(
        [
            transforms.Resize(
                (image_size, image_size),
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    if static_augmentation:
        return RecordAwareCompose(transform_steps)
    return transforms.Compose(transform_steps)