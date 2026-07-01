from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from torchvision import transforms
from torchvision.transforms import InterpolationMode

from micv.transforms.compose import RecordAwareCompose
from micv.transforms.defaults import DEFAULT_MEAN, DEFAULT_STD
from micv.transforms.policies import NTIREAugmentationPolicy, StaticNTIREValidationPolicy


@dataclass(frozen=True)
class TransformPolicyStage:
    enabled: bool = True
    clean_prob: float = 0.30
    max_ops: int = 5
    severity: str = "mixed"
    op_pool: str | None = None
    intensity: str | None = None


def build_train_transform(
    image_size: int = 512,
    difficulty: str = "mixed",
    clean_prob: float = 0.30,
    max_ops: int = 5,
    mean: Sequence[float] = DEFAULT_MEAN,
    std: Sequence[float] = DEFAULT_STD,
    pre_crop: TransformPolicyStage | Mapping[str, Any] | Any | None = None,
    post_crop: TransformPolicyStage | Mapping[str, Any] | Any | None = None,
) -> transforms.Compose:
    pre_crop_stage = _coerce_stage(
        pre_crop,
        TransformPolicyStage(clean_prob=0.60, max_ops=2, severity="train"),
    )
    post_crop_stage = _coerce_stage(
        post_crop,
        TransformPolicyStage(clean_prob=clean_prob, max_ops=max_ops, severity=difficulty),
    )

    transform_steps: list[object] = []
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
    return transforms.Compose(transform_steps)


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
) -> transforms.Compose | RecordAwareCompose:
    transform_steps: list[object] = []
    if static_augmentation:
        transform_steps.append(StaticNTIREValidationPolicy())
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