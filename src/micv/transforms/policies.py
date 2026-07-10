from __future__ import annotations

import importlib
import random
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
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


# Nyquist notch: removes 2px grid artifacts (binomial * (-1)^n, +1 at center).
NYQUIST_NOTCH_KERNEL = (
    np.array([-1.0, 6.0, -15.0, 20.0, -15.0, 6.0, -1.0], dtype=np.float32) / 64.0
)


def _convolve_axis(array: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    radius = kernel.size // 2
    pad_widths = [(0, 0)] * array.ndim
    pad_widths[axis] = (radius, radius)
    padded = np.pad(array, pad_widths, mode="edge")
    length = array.shape[axis]
    out = np.zeros_like(array)
    for offset, weight in enumerate(kernel):
        window = [slice(None)] * array.ndim
        window[axis] = slice(offset, offset + length)
        out += weight * padded[tuple(window)]
    return out


def nyquist_notch(image: Image.Image) -> Image.Image:
    """Cancel alternating-pixel (2px grid) flicker via a separable notch filter.

    Port of the GLSL shader: ``out = center - Bx - By + Bxy`` where ``Bx``/``By``
    are 1D convolutions with :data:`NYQUIST_NOTCH_KERNEL` along width/height and
    ``Bxy`` is the separable 2D response. Edge sampling matches GLSL
    clamp-to-edge.
    """
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    bx = _convolve_axis(array, NYQUIST_NOTCH_KERNEL, axis=1)
    by = _convolve_axis(array, NYQUIST_NOTCH_KERNEL, axis=0)
    bxy = _convolve_axis(bx, NYQUIST_NOTCH_KERNEL, axis=0)
    notched = array - bx - by + bxy
    return Image.fromarray(
        np.clip(notched * 255.0, 0.0, 255.0).astype(np.uint8)
    )


def _normalize_generator_name(value: Any) -> str:
    text = str(value).lower()
    for separator in ("-", "_", " "):
        text = text.replace(separator, "")
    return text


class GeneratorConditionalNyquistNotch:
    """Apply :func:`nyquist_notch` to records from specific generators.

    Matching is exact on normalized names (lowercase, ``-``/``_``/spaces
    stripped), so e.g. ``Qwen-Image`` matches ``qwenimage`` but ``animagine``
    does not match ``anima``.
    """

    needs_record = True

    DEFAULT_GENERATORS = frozenset({"qwen", "qwenimage", "anima", "krea2"})

    def __init__(
        self,
        probability: float = 0.35,
        generators: Iterable[str] | None = None,
    ) -> None:
        self.probability = probability
        self.generators = frozenset(
            _normalize_generator_name(name)
            for name in (self.DEFAULT_GENERATORS if generators is None else generators)
        )

    def __call__(self, image: Image.Image, record: Any = None) -> Image.Image:
        if not self._matches(record):
            return image
        if random.random() >= self.probability:
            return image
        return nyquist_notch(image)

    def _matches(self, record: Any) -> bool:
        metadata = getattr(record, "metadata", None)
        if not isinstance(metadata, dict):
            return False
        generator = metadata.get("generator")
        if generator in {None, ""}:
            return False
        return _normalize_generator_name(generator) in self.generators


class StaticNTIREValidationPolicy:
    needs_record = True

    def __init__(self, policy_version: str | None = None, severity: str = "hard") -> None:
        self.severity = severity
        self.policy_version = policy_version or f"val_{severity}_v1"

    def __call__(self, image: Image.Image, record: Any) -> Image.Image:
        seed = stable_int_hash(f"{self.policy_version}:{record_seed_identity(record)}")
        state = random.getstate()
        try:
            random.seed(seed)
            return NTIREAugmentationPolicy(
                clean_prob=0.0,
                max_ops=random.randint(1, 5),
                severity=self.severity,
            )(image)
        finally:
            random.setstate(state)


def _compat_transforms_module() -> Any:
    return importlib.import_module("micv.data.transforms")