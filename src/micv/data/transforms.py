from __future__ import annotations

import hashlib
import io
import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as transform_functional

DEFAULT_MEAN = (0.485, 0.456, 0.406)
DEFAULT_STD = (0.229, 0.224, 0.225)

AUG_GROUPS = {
    "blur": [
        "gaussian_blur",
        "lens_blur",
        "motion_blur",
        "glass_blur",
    ],
    "compression": [
        "jpeg",
        "jpeg2000",
        "multi_jpeg",
        "jpeg_then_jpeg2000",
        "neural_compression_proxy",
    ],
    "noise": [
        "white_noise",
        "impulse_noise",
        "multiplicative_noise",
        "iso_noise",
        "shot_noise",
    ],
    "color_tone": [
        "color_shift",
        "color_saturation",
        "brightness_up",
        "brightness_down",
        "color_jitter",
        "color_quantization",
        "linear_contrast",
        "rgb_channel_shift",
        "random_tone_curve",
        "clahe",
    ],
    "geometry": [
        "random_crop",
        "random_aspect_crop",
        "downscale",
        "pixelation",
        "perspective",
    ],
    "watermark": [
        "invisible_watermark_proxy",
        "watermark_attack_proxy",
    ],
}

TRAIN_OPS = {
    "gaussian_blur",
    "lens_blur",
    "color_shift",
    "color_saturation",
    "jpeg",
    "white_noise",
    "impulse_noise",
    "brightness_up",
    "brightness_down",
    "color_jitter",
    "color_quantization",
    "linear_contrast",
}

VAL_EXTRA_OPS = {
    "motion_blur",
    "multiplicative_noise",
    "pixelation",
    "rgb_channel_shift",
    "random_crop",
    "random_aspect_crop",
    "downscale",
}


@dataclass
class NTIREAugmentationPolicy:
    clean_prob: float = 0.30
    max_ops: int = 5
    severity: str = "mixed"

    def __call__(self, image: Image.Image) -> Image.Image:
        if self.severity in {"none", "off", "disabled"} or random.random() < self.clean_prob:
            return image
        available_groups = [group for group in AUG_GROUPS if self._ops_for_group(group)]
        if not available_groups:
            return image
        num_ops = random.randint(1, max(1, self.max_ops))
        groups = random.sample(available_groups, k=min(num_ops, len(available_groups)))
        out = image.convert("RGB")
        for group in groups:
            op_name = random.choice(self._ops_for_group(group))
            out = apply_op(op_name, out, severity=self.severity)
        return out

    def _ops_for_group(self, group: str) -> list[str]:
        ops = AUG_GROUPS[group]
        normalized_severity = _normalize_severity(self.severity)
        if normalized_severity == "train":
            return [op for op in ops if op in TRAIN_OPS]
        if normalized_severity == "val":
            return [op for op in ops if op in VAL_EXTRA_OPS or self._is_train_op(op)]
        return ops

    @staticmethod
    def _is_train_op(op: str) -> bool:
        return op in TRAIN_OPS


class StaticNTIREValidationPolicy:
    needs_record = True

    def __init__(self, policy_version: str = "val_hard_v1") -> None:
        self.policy_version = policy_version

    def __call__(self, image: Image.Image, record: Any) -> Image.Image:
        seed = stable_int_hash(f"{self.policy_version}:{record.path}")
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


class RecordAwareCompose:
    needs_record = True

    def __init__(self, transform_steps: Sequence[object]) -> None:
        self.transforms = list(transform_steps)

    def __call__(self, image: Image.Image, record: Any) -> Any:
        out: Any = image
        for transform in self.transforms:
            if getattr(transform, "needs_record", False):
                out = transform(out, record)
            else:
                out = transform(out)
        return out


def build_train_transform(
    image_size: int = 512,
    difficulty: str = "mixed",
    clean_prob: float = 0.30,
    max_ops: int = 5,
    mean: Sequence[float] = DEFAULT_MEAN,
    std: Sequence[float] = DEFAULT_STD,
) -> transforms.Compose:
    return transforms.Compose(
        [
            NTIREAugmentationPolicy(clean_prob=0.60, max_ops=2, severity="train"),
            transforms.RandomResizedCrop(
                image_size,
                scale=(0.65, 1.0),
                ratio=(0.75, 1.333),
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            ),
            NTIREAugmentationPolicy(clean_prob=clean_prob, max_ops=max_ops, severity=difficulty),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
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


def stable_int_hash(value: str) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def apply_op(op_name: str, image: Image.Image, severity: str = "mixed") -> Image.Image:
    effective_severity = _sample_effective_severity(severity)
    operations = {
        "gaussian_blur": _gaussian_blur,
        "lens_blur": _lens_blur,
        "motion_blur": _motion_blur,
        "glass_blur": _glass_blur,
        "jpeg": _jpeg,
        "jpeg2000": _jpeg2000,
        "multi_jpeg": _multi_jpeg,
        "jpeg_then_jpeg2000": _jpeg_then_jpeg2000,
        "neural_compression_proxy": _neural_compression_proxy,
        "white_noise": _white_noise,
        "impulse_noise": _impulse_noise,
        "multiplicative_noise": _multiplicative_noise,
        "iso_noise": _iso_noise,
        "shot_noise": _shot_noise,
        "color_shift": _color_shift,
        "color_saturation": _color_saturation,
        "brightness_up": _brightness_up,
        "brightness_down": _brightness_down,
        "color_jitter": _color_jitter,
        "color_quantization": _color_quantization,
        "linear_contrast": _linear_contrast,
        "rgb_channel_shift": _rgb_channel_shift,
        "random_tone_curve": _random_tone_curve,
        "clahe": _clahe,
        "random_crop": _random_crop,
        "random_aspect_crop": _random_aspect_crop,
        "downscale": _downscale,
        "pixelation": _pixelation,
        "perspective": _perspective,
        "invisible_watermark_proxy": _invisible_watermark_proxy,
        "watermark_attack_proxy": _watermark_attack_proxy,
    }
    try:
        return operations[op_name](image.convert("RGB"), effective_severity)
    except KeyError as error:
        raise ValueError(f"Unknown augmentation operation: {op_name}") from error


def _normalize_severity(severity: str) -> str:
    severity_aliases = {
        "easy": "train",
        "medium": "val",
        "none": "none",
        "off": "none",
        "disabled": "none",
    }
    return severity_aliases.get(severity, severity)


def _sample_effective_severity(severity: str) -> str:
    normalized_severity = _normalize_severity(severity)
    if normalized_severity == "mixed":
        return random.choices(["train", "val", "hard"], weights=[0.4, 0.35, 0.25], k=1)[0]
    if normalized_severity == "test":
        return "hard"
    return normalized_severity


def _pick(severity: str, train: float, val: float, hard: float) -> float:
    values = {"train": train, "val": val, "hard": hard}
    return values.get(severity, hard)


def _uniform(
    severity: str,
    train: tuple[float, float],
    val: tuple[float, float],
    hard: tuple[float, float],
) -> float:
    low, high = {"train": train, "val": val, "hard": hard}.get(severity, hard)
    return random.uniform(low, high)


def _randint(
    severity: str,
    train: tuple[int, int],
    val: tuple[int, int],
    hard: tuple[int, int],
) -> int:
    low, high = {"train": train, "val": val, "hard": hard}.get(severity, hard)
    return random.randint(low, high)


def _rng() -> np.random.Generator:
    return np.random.default_rng(random.randrange(0, 2**32))


def _to_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def _to_image(array: np.ndarray) -> Image.Image:
    return Image.fromarray(np.clip(array * 255.0, 0, 255).astype(np.uint8))


def _resize_back(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    return image.resize(size, Image.Resampling.BICUBIC)


def _gaussian_blur(image: Image.Image, severity: str) -> Image.Image:
    radius = _uniform(severity, (0.2, 0.8), (0.6, 1.5), (1.2, 2.8))
    return image.filter(ImageFilter.GaussianBlur(radius=radius))


def _lens_blur(image: Image.Image, severity: str) -> Image.Image:
    radius = _uniform(severity, (0.4, 1.0), (0.8, 1.8), (1.4, 3.2))
    blurred = image.filter(ImageFilter.BoxBlur(radius=radius))
    sharpness = _uniform(severity, (0.8, 1.0), (0.6, 0.9), (0.4, 0.8))
    return ImageEnhance.Sharpness(blurred).enhance(sharpness)


def _motion_blur(image: Image.Image, severity: str) -> Image.Image:
    desired_kernel_size = _randint(severity, (3, 5), (5, 9), (9, 15))
    desired_kernel_size += 1 - desired_kernel_size % 2
    kernel_size = 5 if desired_kernel_size >= 5 else 3
    passes = max(1, round(desired_kernel_size / kernel_size))
    direction = random.choice(["horizontal", "vertical", "diag_down", "diag_up"])
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    if direction == "horizontal":
        kernel[kernel_size // 2, :] = 1.0
    elif direction == "vertical":
        kernel[:, kernel_size // 2] = 1.0
    elif direction == "diag_down":
        np.fill_diagonal(kernel, 1.0)
    else:
        indices = np.arange(kernel_size)
        kernel[indices, kernel_size - 1 - indices] = 1.0
    kernel = (kernel / kernel.sum()).reshape(-1).tolist()
    motion_filter = ImageFilter.Kernel((kernel_size, kernel_size), kernel, scale=1.0)
    out = image
    for _ in range(passes):
        out = out.filter(motion_filter)
    return out


def _glass_blur(image: Image.Image, severity: str) -> Image.Image:
    blurred = _gaussian_blur(image, severity)
    array = np.asarray(blurred).copy()
    rng = _rng()
    max_shift = _randint(severity, (1, 1), (1, 2), (2, 3))
    for _ in range(_randint(severity, (4, 8), (8, 14), (12, 24))):
        dx = int(rng.integers(-max_shift, max_shift + 1))
        dy = int(rng.integers(-max_shift, max_shift + 1))
        array = np.roll(array, shift=(dy, dx), axis=(0, 1))
    return Image.fromarray(array).filter(ImageFilter.GaussianBlur(radius=0.4))


def _jpeg(image: Image.Image, severity: str) -> Image.Image:
    quality = _randint(severity, (70, 95), (45, 85), (25, 70))
    return _save_reopen(image, "JPEG", quality=quality)


def _jpeg2000(image: Image.Image, severity: str) -> Image.Image:
    compression_rate = _uniform(
        severity,
        (4.0, 10.0),
        (10.0, 24.0),
        (20.0, 45.0),
    )
    try:
        return _save_reopen(
            image,
            "JPEG2000",
            quality_mode="rates",
            quality_layers=[compression_rate],
        )
    except Exception:
        return _jpeg(_downscale(image, severity), severity)


def _multi_jpeg(image: Image.Image, severity: str) -> Image.Image:
    out = image
    for _ in range(_randint(severity, (2, 2), (2, 3), (3, 4))):
        out = _jpeg(out, severity)
    return out


def _jpeg_then_jpeg2000(image: Image.Image, severity: str) -> Image.Image:
    return _jpeg2000(_jpeg(image, severity), severity)


def _neural_compression_proxy(image: Image.Image, severity: str) -> Image.Image:
    compressed = _downscale(image, severity)
    compressed = _color_quantization(compressed, severity)
    return _jpeg(compressed, severity)


def _save_reopen(image: Image.Image, image_format: str, **save_kwargs: Any) -> Image.Image:
    buffer = io.BytesIO()
    image.save(buffer, format=image_format, **save_kwargs)
    buffer.seek(0)
    with Image.open(buffer) as compressed:
        compressed.load()
        return compressed.convert("RGB")


def _white_noise(image: Image.Image, severity: str) -> Image.Image:
    array = _to_array(image)
    sigma = _uniform(severity, (0.006, 0.025), (0.015, 0.045), (0.035, 0.09))
    return _to_image(array + _rng().normal(0.0, sigma, size=array.shape))


def _impulse_noise(image: Image.Image, severity: str) -> Image.Image:
    array = _to_array(image)
    probability = _uniform(
        severity,
        (0.001, 0.006),
        (0.004, 0.015),
        (0.01, 0.035),
    )
    rng = _rng()
    mask = rng.random(array.shape[:2]) < probability
    salt = rng.random(array.shape[:2]) < 0.5
    array[mask & salt] = 1.0
    array[mask & ~salt] = 0.0
    return _to_image(array)


def _multiplicative_noise(image: Image.Image, severity: str) -> Image.Image:
    array = _to_array(image)
    sigma = _uniform(severity, (0.015, 0.04), (0.03, 0.075), (0.06, 0.14))
    return _to_image(array * _rng().normal(1.0, sigma, size=array.shape))


def _iso_noise(image: Image.Image, severity: str) -> Image.Image:
    array = _to_array(image)
    rng = _rng()
    luminance_sigma = _uniform(
        severity,
        (0.008, 0.025),
        (0.018, 0.055),
        (0.04, 0.11),
    )
    chroma_sigma = luminance_sigma * 0.45
    luminance_noise = rng.normal(0.0, luminance_sigma, size=array.shape[:2])[..., None]
    chroma_noise = rng.normal(0.0, chroma_sigma, size=array.shape)
    return _to_image(array + luminance_noise + chroma_noise)


def _shot_noise(image: Image.Image, severity: str) -> Image.Image:
    array = _to_array(image)
    scale = _pick(severity, 95.0, 55.0, 28.0)
    return _to_image(_rng().poisson(array * scale) / scale)


def _color_shift(image: Image.Image, severity: str) -> Image.Image:
    hsv = np.asarray(image.convert("HSV")).copy()
    hue_shift = _randint(severity, (-5, 5), (-10, 10), (-18, 18))
    hsv[..., 0] = (hsv[..., 0].astype(np.int16) + hue_shift) % 256
    return Image.fromarray(hsv.astype(np.uint8), mode="HSV").convert("RGB")


def _color_saturation(image: Image.Image, severity: str) -> Image.Image:
    factor = _uniform(severity, (0.85, 1.18), (0.7, 1.35), (0.5, 1.6))
    return ImageEnhance.Color(image).enhance(factor)


def _brightness_up(image: Image.Image, severity: str) -> Image.Image:
    factor = _uniform(severity, (1.03, 1.12), (1.08, 1.24), (1.18, 1.45))
    return ImageEnhance.Brightness(image).enhance(factor)


def _brightness_down(image: Image.Image, severity: str) -> Image.Image:
    factor = _uniform(severity, (0.88, 0.97), (0.72, 0.92), (0.5, 0.82))
    return ImageEnhance.Brightness(image).enhance(factor)


def _color_jitter(image: Image.Image, severity: str) -> Image.Image:
    out = image
    enhancers = [
        ImageEnhance.Brightness,
        ImageEnhance.Contrast,
        ImageEnhance.Color,
        ImageEnhance.Sharpness,
    ]
    for enhancer in random.sample(enhancers, k=random.randint(2, len(enhancers))):
        factor = _uniform(severity, (0.9, 1.1), (0.75, 1.25), (0.55, 1.5))
        out = enhancer(out).enhance(factor)
    return out


def _color_quantization(image: Image.Image, severity: str) -> Image.Image:
    bits = _randint(severity, (5, 7), (4, 6), (3, 5))
    return ImageOps.posterize(image, bits=bits)


def _linear_contrast(image: Image.Image, severity: str) -> Image.Image:
    factor = _uniform(severity, (0.85, 1.2), (0.7, 1.4), (0.5, 1.7))
    return ImageEnhance.Contrast(image).enhance(factor)


def _rgb_channel_shift(image: Image.Image, severity: str) -> Image.Image:
    array = np.asarray(image).copy()
    max_shift = _randint(severity, (1, 3), (2, 6), (4, 12))
    for channel in range(3):
        dx = random.randint(-max_shift, max_shift)
        dy = random.randint(-max_shift, max_shift)
        array[..., channel] = np.roll(array[..., channel], shift=(dy, dx), axis=(0, 1))
    return Image.fromarray(array)


def _random_tone_curve(image: Image.Image, severity: str) -> Image.Image:
    gamma = _uniform(severity, (0.9, 1.1), (0.75, 1.3), (0.55, 1.65))
    lift = _uniform(severity, (-0.02, 0.02), (-0.05, 0.05), (-0.09, 0.09))
    lookup = [
        int(np.clip(((value / 255.0) ** gamma + lift) * 255.0, 0, 255))
        for value in range(256)
    ]
    return image.point(lookup * 3)


def _clahe(image: Image.Image, severity: str) -> Image.Image:
    autocontrast = ImageOps.autocontrast(
        image,
        cutoff=_uniform(severity, (0.5, 1.5), (1.0, 2.5), (2.0, 4.0)),
    )
    equalized = ImageOps.equalize(autocontrast)
    alpha = _uniform(severity, (0.1, 0.25), (0.2, 0.4), (0.35, 0.65))
    return Image.blend(autocontrast, equalized, alpha=alpha)


def _random_crop(image: Image.Image, severity: str) -> Image.Image:
    width, height = image.size
    scale = _uniform(severity, (0.86, 0.98), (0.72, 0.92), (0.55, 0.85))
    crop_width = max(1, int(width * scale))
    crop_height = max(1, int(height * scale))
    left = random.randint(0, max(0, width - crop_width))
    top = random.randint(0, max(0, height - crop_height))
    crop = image.crop((left, top, left + crop_width, top + crop_height))
    return _resize_back(crop, image.size)


def _random_aspect_crop(image: Image.Image, severity: str) -> Image.Image:
    width, height = image.size
    area_scale = _uniform(severity, (0.84, 0.98), (0.68, 0.92), (0.5, 0.85))
    aspect = _uniform(
        severity,
        (0.9, 1.12),
        (0.75, 1.35),
        (0.55, 1.7),
    ) * (width / height)
    crop_width = min(
        width,
        max(1, int((width * height * area_scale * aspect) ** 0.5)),
    )
    crop_height = min(height, max(1, int(crop_width / aspect)))
    left = random.randint(0, max(0, width - crop_width))
    top = random.randint(0, max(0, height - crop_height))
    crop = image.crop((left, top, left + crop_width, top + crop_height))
    return _resize_back(crop, image.size)


def _downscale(image: Image.Image, severity: str) -> Image.Image:
    width, height = image.size
    scale = _uniform(severity, (0.65, 0.9), (0.45, 0.75), (0.25, 0.6))
    small_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    interpolation = random.choice(
        [
            Image.Resampling.BILINEAR,
            Image.Resampling.BICUBIC,
            Image.Resampling.LANCZOS,
        ]
    )
    return image.resize(small_size, interpolation).resize((width, height), interpolation)


def _pixelation(image: Image.Image, severity: str) -> Image.Image:
    width, height = image.size
    scale = _uniform(severity, (0.35, 0.55), (0.18, 0.38), (0.08, 0.22))
    small_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return image.resize(small_size, Image.Resampling.BILINEAR).resize(
        image.size,
        Image.Resampling.NEAREST,
    )


def _perspective(image: Image.Image, severity: str) -> Image.Image:
    width, height = image.size
    distortion = _uniform(severity, (0.04, 0.10), (0.08, 0.18), (0.15, 0.3))
    max_dx = int(width * distortion)
    max_dy = int(height * distortion)
    startpoints = [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]
    endpoints = [
        [random.randint(0, max_dx), random.randint(0, max_dy)],
        [
            width - 1 - random.randint(0, max_dx),
            random.randint(0, max_dy),
        ],
        [
            width - 1 - random.randint(0, max_dx),
            height - 1 - random.randint(0, max_dy),
        ],
        [
            random.randint(0, max_dx),
            height - 1 - random.randint(0, max_dy),
        ],
    ]
    return transform_functional.perspective(
        image,
        startpoints=startpoints,
        endpoints=endpoints,
        interpolation=InterpolationMode.BICUBIC,
        fill=0,
    )


def _invisible_watermark_proxy(image: Image.Image, severity: str) -> Image.Image:
    array = _to_array(image)
    amplitude = _uniform(severity, (0.002, 0.006), (0.004, 0.01), (0.008, 0.018))
    y_coords, x_coords = np.indices(array.shape[:2])
    frequency = _uniform(severity, (0.04, 0.08), (0.06, 0.12), (0.1, 0.2))
    pattern = np.sin((x_coords + y_coords) * frequency)[..., None] * amplitude
    return _to_image(array + pattern)


def _watermark_attack_proxy(image: Image.Image, severity: str) -> Image.Image:
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    width, height = image.size
    alpha = _randint(severity, (18, 35), (28, 55), (45, 85))
    label = random.choice(["MICV", "SAMPLE", "AIGC", "VERIFY"])
    spacing = max(24, min(width, height) // random.randint(4, 7))
    for y in range(-spacing, height + spacing, spacing):
        for x in range(-spacing, width + spacing, spacing * 2):
            draw.text((x, y), label, fill=(255, 255, 255, alpha))
    return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")