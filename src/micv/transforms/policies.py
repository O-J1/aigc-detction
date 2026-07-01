from __future__ import annotations

import importlib
import random
from dataclasses import dataclass
from typing import Any

from PIL import Image

from micv.transforms.identity import record_seed_identity, stable_int_hash


@dataclass
class NTIREAugmentationPolicy:
    clean_prob: float = 0.30
    max_ops: int = 5
    severity: str = "mixed"
    op_pool: str | None = None
    intensity: str | None = None

    def __call__(self, image: Image.Image) -> Image.Image:
        transforms_module = _compat_transforms_module()
        intensity = self.intensity or self.severity
        if intensity in {"none", "off", "disabled"} or random.random() < self.clean_prob:
            return image
        available_groups = [group for group in transforms_module.AUG_GROUPS if self._ops_for_group(group)]
        if not available_groups:
            return image
        num_ops = random.randint(1, max(1, self.max_ops))
        groups = random.sample(available_groups, k=min(num_ops, len(available_groups)))
        out = image.convert("RGB")
        for group in groups:
            effective_severity = transforms_module._sample_effective_severity(intensity)
            ops = self._ops_for_group(group, severity_override=effective_severity)
            if not ops:
                continue
            op_name = random.choice(ops)
            out = transforms_module.apply_op(op_name, out, severity=effective_severity)
        return out

    def _ops_for_group(
        self,
        group: str,
        severity_override: str | None = None,
    ) -> list[str]:
        transforms_module = _compat_transforms_module()
        ops = transforms_module.AUG_GROUPS[group]
        op_pool = self.op_pool or severity_override or self.severity
        normalized_pool = transforms_module._normalize_severity(op_pool)
        if normalized_pool == "train":
            return [op for op in ops if op in transforms_module.TRAIN_OPS]
        if normalized_pool == "val":
            return [op for op in ops if op in transforms_module.VAL_EXTRA_OPS or self._is_train_op(op)]
        return ops

    @staticmethod
    def _is_train_op(op: str) -> bool:
        return op in _compat_transforms_module().TRAIN_OPS


class StaticNTIREValidationPolicy:
    needs_record = True

    def __init__(self, policy_version: str = "val_hard_v1") -> None:
        self.policy_version = policy_version

    def __call__(self, image: Image.Image, record: Any) -> Image.Image:
        seed = stable_int_hash(f"{self.policy_version}:{record_seed_identity(record)}")
        state = random.getstate()
        try:
            random.seed(seed)
            return NTIREAugmentationPolicy(
                clean_prob=0.0,
                max_ops=random.randint(1, 5),
                severity="hard",
            )(image)
        finally:
            random.setstate(state)


def _compat_transforms_module() -> Any:
    return importlib.import_module("micv.data.transforms")