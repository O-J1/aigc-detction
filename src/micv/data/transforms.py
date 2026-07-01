from __future__ import annotations

import importlib
import io
import random
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as transform_functional

from micv.transforms.builders import TransformPolicyStage, build_eval_transform, build_train_transform
from micv.transforms.compose import RecordAwareCompose
from micv.transforms.defaults import DEFAULT_MEAN, DEFAULT_STD
from micv.transforms.identity import record_seed_identity as _record_seed_identity
from micv.transforms.identity import stable_int_hash
from micv.transforms.policies import NTIREAugmentationPolicy, StaticNTIREValidationPolicy
from micv.transforms.registry import AUG_GROUPS, OPERATIONS, TRAIN_OPS, VAL_EXTRA_OPS, Operation, apply_op
from micv.transforms.severity import normalize_severity as _normalize_severity
from micv.transforms.severity import sample_effective_severity as _sample_effective_severity

__all__ = [
    "AUG_GROUPS",
    "DEFAULT_MEAN",
    "DEFAULT_STD",
    "NTIREAugmentationPolicy",
    "OPERATIONS",
    "Operation",
    "RecordAwareCompose",
    "StaticNTIREValidationPolicy",
    "TRAIN_OPS",
    "TransformPolicyStage",
    "VAL_EXTRA_OPS",
    "_normalize_severity",
    "_record_seed_identity",
    "_sample_effective_severity",
    "apply_op",
    "build_eval_transform",
    "build_train_transform",
    "stable_int_hash",
]

_HEIF_REGISTERED = False
_HEIF_REGISTER_ATTEMPTED = False
_JXL_IMPORT_ATTEMPTED = False


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


def _webp(image: Image.Image, severity: str) -> Image.Image:
    quality = _randint(severity, (72, 96), (48, 88), (28, 72))
    method = _randint(severity, (3, 6), (2, 6), (0, 6))
    return _codec_or_fallback(
        image,
        severity,
        "WEBP",
        _jpeg,
        quality=quality,
        method=method,
    )


def _avif(image: Image.Image, severity: str) -> Image.Image:
    quality = _randint(severity, (65, 95), (42, 82), (22, 64))
    speed = _randint(severity, (6, 10), (4, 8), (2, 7))
    subsampling = random.choices(
        ["4:2:0", "4:2:2", "4:4:4"],
        weights=[0.70, 0.20, 0.10],
        k=1,
    )[0]
    return _codec_or_fallback(
        image,
        severity,
        "AVIF",
        _webp,
        quality=quality,
        speed=speed,
        subsampling=subsampling,
    )


def _jxl(image: Image.Image, severity: str) -> Image.Image:
    quality = _randint(severity, (78, 99), (55, 90), (35, 78))
    return _codec_or_fallback(
        image,
        severity,
        "JXL",
        _webp,
        quality=quality,
    )


def _multi_jpeg(image: Image.Image, severity: str) -> Image.Image:
    out = image
    for _ in range(_randint(severity, (2, 2), (2, 3), (3, 4))):
        out = _jpeg(out, severity)
    return out


def _jpeg_then_jpeg2000(image: Image.Image, severity: str) -> Image.Image:
    return _jpeg2000(_jpeg(image, severity), severity)


def _resize_jitter(image: Image.Image, severity: str) -> Image.Image:
    width, height = image.size
    scale = _uniform(severity, (0.85, 1.08), (0.65, 1.12), (0.45, 1.20))
    new_size = (max(8, int(width * scale)), max(8, int(height * scale)))
    down_interpolation = random.choice(
        [
            Image.Resampling.BILINEAR,
            Image.Resampling.BICUBIC,
            Image.Resampling.LANCZOS,
        ]
    )
    up_interpolation = random.choice(
        [
            Image.Resampling.BILINEAR,
            Image.Resampling.BICUBIC,
            Image.Resampling.LANCZOS,
        ]
    )
    return image.resize(new_size, down_interpolation).resize((width, height), up_interpolation)


def _multi_codec(image: Image.Image, severity: str) -> Image.Image:
    out = image
    num_rounds = _randint(severity, (2, 2), (2, 3), (3, 5))
    codec_ops = [
        (_jpeg, 0.38),
        (_webp, 0.24),
        (_avif, 0.16),
        (_jpeg2000, 0.12),
        (_jxl, 0.10),
    ]
    codec_funcs, codec_weights = zip(*codec_ops, strict=True)

    for _ in range(num_rounds):
        if random.random() < _pick(severity, 0.25, 0.45, 0.65):
            out = _resize_jitter(out, severity)
        codec_func = random.choices(codec_funcs, weights=codec_weights, k=1)[0]
        out = codec_func(out, severity)

    return out


def _resize_then_codec(image: Image.Image, severity: str) -> Image.Image:
    out = _resize_jitter(image, severity)
    codec_func = random.choices(
        [_jpeg, _webp, _avif, _jpeg2000],
        weights=[0.45, 0.28, 0.17, 0.10],
        k=1,
    )[0]
    return codec_func(out, severity)


def _save_reopen(image: Image.Image, image_format: str, **save_kwargs: Any) -> Image.Image:
    buffer = io.BytesIO()
    image.save(buffer, format=image_format, **save_kwargs)
    buffer.seek(0)
    with Image.open(buffer) as compressed:
        compressed.load()
        return compressed.convert("RGB")


def _pil_can_save(image_format: str) -> bool:
    _ensure_codec_registered(image_format)
    Image.init()
    return image_format.upper() in Image.SAVE


def _ensure_codec_registered(image_format: str) -> None:
    global _HEIF_REGISTERED, _HEIF_REGISTER_ATTEMPTED, _JXL_IMPORT_ATTEMPTED

    normalized_format = image_format.upper()
    if normalized_format in {"AVIF", "HEIF"} and not _HEIF_REGISTER_ATTEMPTED:
        _HEIF_REGISTER_ATTEMPTED = True
        try:
            pillow_heif = importlib.import_module("pillow_heif")
        except ImportError:
            return
        pillow_heif.register_heif_opener()
        _HEIF_REGISTERED = True

    if normalized_format == "JXL" and not _JXL_IMPORT_ATTEMPTED:
        _JXL_IMPORT_ATTEMPTED = True
        try:
            importlib.import_module("pillow_jxl")
        except ImportError:
            return


def _save_reopen_or_none(
    image: Image.Image,
    image_format: str,
    **save_kwargs: Any,
) -> Image.Image | None:
    try:
        if not _pil_can_save(image_format):
            return None
        return _save_reopen(image, image_format, **save_kwargs)
    except Exception:
        return None


def _codec_or_fallback(
    image: Image.Image,
    severity: str,
    image_format: str,
    fallback: Any,
    **save_kwargs: Any,
) -> Image.Image:
    out = _save_reopen_or_none(image, image_format, **save_kwargs)
    if out is not None:
        return out
    return fallback(image, severity)


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


def _limit_long_edge(image: Image.Image, max_side: int) -> Image.Image:
    width, height = image.size
    long_edge = max(width, height)
    if long_edge <= max_side:
        return image

    scale = max_side / long_edge
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    interpolation = random.choice(
        [
            Image.Resampling.BILINEAR,
            Image.Resampling.BICUBIC,
            Image.Resampling.LANCZOS,
        ]
    )
    return image.resize(new_size, interpolation)


def _platform_recompress(image: Image.Image, severity: str) -> Image.Image:
    original_size = image.size
    max_side = random.choice(
        {
            "train": [2048, 1920, 1600],
            "val": [1920, 1600, 1280],
            "hard": [1600, 1280, 1024, 960],
        }.get(severity, [1600, 1280, 1024])
    )
    out = _limit_long_edge(image, max_side)
    codec_func = random.choices(
        [_jpeg, _webp, _avif],
        weights=[0.55, 0.30, 0.15],
        k=1,
    )[0]
    out = codec_func(out, severity)
    return _resize_back(out, original_size)


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


def _screenshot_resample(image: Image.Image, severity: str) -> Image.Image:
    out = _resize_jitter(image, severity)
    if random.random() < _pick(severity, 0.25, 0.40, 0.60):
        out = ImageEnhance.Sharpness(out).enhance(
            _uniform(severity, (1.02, 1.12), (1.05, 1.25), (1.10, 1.45))
        )
    if random.random() < _pick(severity, 0.20, 0.35, 0.50):
        out = _linear_contrast(out, severity)
    if random.random() < _pick(severity, 0.20, 0.35, 0.50):
        out = _jpeg(out, severity)
    return out


def _letterbox_pad(image: Image.Image, severity: str) -> Image.Image:
    width, height = image.size
    pad_fraction = _uniform(severity, (0.01, 0.04), (0.03, 0.08), (0.06, 0.14))
    pad_width = int(width * pad_fraction)
    pad_height = int(height * pad_fraction)
    fill = random.choice(
        [
            (0, 0, 0),
            (255, 255, 255),
            (245, 245, 245),
            (16, 16, 16),
        ]
    )
    padded = ImageOps.expand(
        image,
        border=(pad_width, pad_height, pad_width, pad_height),
        fill=fill,
    )
    return _resize_back(padded, image.size)


def _subtle_watermark(image: Image.Image, severity: str) -> Image.Image:
    width, height = image.size
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    alpha = _randint(severity, (18, 34), (28, 55), (45, 85))
    box_width = max(
        1,
        int(width * _uniform(severity, (0.12, 0.22), (0.16, 0.30), (0.22, 0.40))),
    )
    box_height = max(
        1,
        int(height * _uniform(severity, (0.035, 0.07), (0.05, 0.10), (0.07, 0.14))),
    )
    margin = int(min(width, height) * 0.035)
    positions = [
        (margin, margin),
        (width - box_width - margin, margin),
        (margin, height - box_height - margin),
        (width - box_width - margin, height - box_height - margin),
    ]
    left, top = random.choice(positions)
    fill = random.choice(
        [
            (255, 255, 255, alpha),
            (0, 0, 0, alpha),
        ]
    )
    draw.rounded_rectangle(
        (left, top, left + box_width, top + box_height),
        radius=max(2, box_height // 5),
        fill=fill,
    )
    return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")


def _print_scan(image: Image.Image, severity: str) -> Image.Image:
    out = image
    if random.random() < 0.75:
        out = _perspective(out, severity)
    out = _gaussian_blur(out, severity)
    if random.random() < 0.65:
        out = _linear_contrast(out, severity)
    if random.random() < 0.55:
        out = _iso_noise(out, severity)
    if random.random() < 0.75:
        out = _jpeg(out, severity)
    return out


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
    perspective = getattr(transform_functional, "perspective")
    return perspective(
        image,
        startpoints=startpoints,
        endpoints=endpoints,
        interpolation=InterpolationMode.BICUBIC,
        fill=0,
    )
    

SCREEN_PHOTO_PROFILE_WEIGHTS = {
    "train": {
        "mild": 0.99,
        "medium": 0.01,
        "extra_hard": 0.00,
    },
    "val": {
        "mild": 0.84,
        "medium": 0.15,
        "extra_hard": 0.01,
    },
    "hard": {
        "mild": 0.55,
        "medium": 0.37,
        "extra_hard": 0.08,
    },
}

def _sample_screen_photo_profile(severity: str) -> str:
    effective_severity = _sample_effective_severity(severity)

    weights = SCREEN_PHOTO_PROFILE_WEIGHTS.get(
        effective_severity,
        SCREEN_PHOTO_PROFILE_WEIGHTS["hard"],
    )

    profiles = list(weights.keys())
    probabilities = list(weights.values())

    return random.choices(profiles, weights=probabilities, k=1)[0]


def _screen_photo_profile_to_severity(profile: str) -> str:
    if profile == "mild":
        return "train"
    if profile == "medium":
        return "val"
    return "hard"


def _screen_profile_pick(profile: str, mild: Any, medium: Any, extra_hard: Any) -> Any:
    values = {
        "mild": mild,
        "medium": medium,
        "extra_hard": extra_hard,
    }
    return values.get(profile, medium)

def _camera_screen_photo(image: Image.Image, severity: str) -> Image.Image:
    """Screen-photo simulation with stochastic mild/medium/extra-hard profiles.

    Most training/validation samples should remain readable. The strong
    paper-style moiré/camera artifacts are still reachable, but not every
    sample gets the full pipeline.
    """
    profile = _sample_screen_photo_profile(severity)
    return _camera_screen_photo_with_profile(image, profile)

def _screen_projective_transform_soft(image: Image.Image, profile: str) -> Image.Image:
    width, height = image.size

    distortion = random.uniform(
        *_screen_profile_pick(
            profile,
            (0.002, 0.008),
            (0.006, 0.018),
            (0.012, 0.040),
        )
    )

    max_dx = max(1, int(width * distortion))
    max_dy = max(1, int(height * distortion))

    startpoints = [
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1],
    ]

    endpoints = [
        [random.randint(0, max_dx), random.randint(0, max_dy)],
        [width - 1 - random.randint(0, max_dx), random.randint(0, max_dy)],
        [
            width - 1 - random.randint(0, max_dx),
            height - 1 - random.randint(0, max_dy),
        ],
        [random.randint(0, max_dx), height - 1 - random.randint(0, max_dy)],
    ]

    return transform_functional.perspective(
        image,
        startpoints=startpoints,
        endpoints=endpoints,
        interpolation=InterpolationMode.BICUBIC,
        fill=0,
    )


def _screen_optical_blur_profiled(image: Image.Image, profile: str) -> Image.Image:
    radius = random.uniform(
        *_screen_profile_pick(
            profile,
            (0.05, 0.18),
            (0.12, 0.35),
            (0.25, 0.60),
        )
    )

    if radius <= 0.06:
        return image

    return image.filter(ImageFilter.GaussianBlur(radius=radius))


def _screen_capture_noise_profiled(image: Image.Image, profile: str) -> Image.Image:
    array = _to_array(image)
    rng = _rng()

    luminance_sigma = random.uniform(
        *_screen_profile_pick(
            profile,
            (0.0010, 0.0040),
            (0.0025, 0.0080),
            (0.0050, 0.0140),
        )
    )

    chroma_sigma = luminance_sigma * random.uniform(0.25, 0.45)

    luminance_noise = rng.normal(
        0.0,
        luminance_sigma,
        size=array.shape[:2],
    )[..., None]

    chroma_noise = rng.normal(
        0.0,
        chroma_sigma,
        size=array.shape,
    )

    return _to_image(array + luminance_noise + chroma_noise)


def _screen_fine_lcd_texture_overlay(image: Image.Image, profile: str) -> Image.Image:
    """Fine vertical LCD/subpixel texture.

    This replaces the destructive full LCD RGB mosaic for normal samples.
    It keeps the image readable while adding visible screen structure.
    """
    array = _to_array(image)
    height, width = array.shape[:2]

    x = np.arange(width, dtype=np.float32)

    period = random.uniform(
        *_screen_profile_pick(
            profile,
            (2.2, 4.4),
            (1.7, 3.7),
            (1.25, 3.1),
        )
    )

    phase = random.uniform(0.0, 2.0 * float(np.pi))
    base = 2.0 * np.pi * x / period + phase

    luma_amp = random.uniform(
        *_screen_profile_pick(
            profile,
            (0.018, 0.045),
            (0.035, 0.075),
            (0.050, 0.110),
        )
    )

    chroma_amp = random.uniform(
        *_screen_profile_pick(
            profile,
            (0.004, 0.012),
            (0.008, 0.024),
            (0.014, 0.040),
        )
    )

    # Vertical luminance modulation.
    vertical = np.sin(base)[None, :, None]
    out = array * (1.0 + luma_amp * vertical)

    # RGB phase-shifted subpixel tint. This gives screen texture without
    # turning every third column into a pure RGB stripe.
    rgb_pattern = np.stack(
        [
            np.sin(base),
            np.sin(base + 2.0 * np.pi / 3.0),
            np.sin(base + 4.0 * np.pi / 3.0),
        ],
        axis=-1,
    )[None, :, :]

    luminance = array.mean(axis=2, keepdims=True)
    out = out + chroma_amp * (0.35 + 0.65 * luminance) * rgb_pattern

    # Very faint horizontal row structure.
    if random.random() < _screen_profile_pick(profile, 0.30, 0.45, 0.65):
        y = np.arange(height, dtype=np.float32)
        row_period = random.uniform(
            *_screen_profile_pick(
                profile,
                (2.5, 5.5),
                (2.0, 5.0),
                (1.6, 4.4),
            )
        )
        row_phase = random.uniform(0.0, 2.0 * float(np.pi))
        row_amp = random.uniform(
            *_screen_profile_pick(
                profile,
                (0.003, 0.010),
                (0.006, 0.018),
                (0.010, 0.030),
            )
        )
        row = np.sin(2.0 * np.pi * y / row_period + row_phase)[:, None, None]
        out = out * (1.0 + row_amp * row)

    return _to_image(out)


def _screen_soft_coarse_moire(image: Image.Image, profile: str) -> Image.Image:
    """Low-amplitude coarse moiré.

    This should look like the real screen photo: visible, but not dominating
    the image.
    """
    array = _to_array(image)
    height, width = array.shape[:2]

    yy, xx = np.meshgrid(
        np.linspace(-1.0, 1.0, height, dtype=np.float32),
        np.linspace(-1.0, 1.0, width, dtype=np.float32),
        indexing="ij",
    )

    theta = random.uniform(0.0, float(np.pi))
    coord = xx * np.cos(theta) + yy * np.sin(theta)

    curvature = random.uniform(
        *_screen_profile_pick(
            profile,
            (-0.08, 0.08),
            (-0.18, 0.18),
            (-0.35, 0.35),
        )
    )
    coord = coord + curvature * (xx * xx + yy * yy)

    frequency = random.uniform(
        *_screen_profile_pick(
            profile,
            (1.2, 4.0),
            (2.0, 7.0),
            (3.5, 12.0),
        )
    )

    phase = random.uniform(0.0, 2.0 * float(np.pi))
    pattern = np.sin(2.0 * np.pi * frequency * coord + phase)[..., None]

    luma_amp = random.uniform(
        *_screen_profile_pick(
            profile,
            (0.006, 0.018),
            (0.012, 0.034),
            (0.020, 0.055),
        )
    )

    chroma_amp = random.uniform(
        *_screen_profile_pick(
            profile,
            (0.002, 0.008),
            (0.004, 0.014),
            (0.008, 0.026),
        )
    )

    color_axis = np.asarray(
        random.choice(
            [
                [0.55, -0.20, -0.35],
                [-0.25, 0.55, -0.30],
                [-0.25, -0.20, 0.45],
                [0.35, 0.10, -0.45],
            ]
        ),
        dtype=np.float32,
    )

    luminance = array.mean(axis=2, keepdims=True)

    out = array * (1.0 + luma_amp * pattern)
    out = out + chroma_amp * (0.25 + 0.75 * luminance) * pattern * color_axis

    return _to_image(out)


def _screen_subtle_rolling_band(image: Image.Image, profile: str) -> Image.Image:
    array = _to_array(image)
    height = array.shape[0]

    y = np.linspace(0.0, 1.0, height, dtype=np.float32)

    frequency = random.uniform(
        *_screen_profile_pick(
            profile,
            (0.7, 2.2),
            (1.2, 3.8),
            (2.0, 6.5),
        )
    )

    phase = random.uniform(0.0, 2.0 * float(np.pi))

    amplitude = random.uniform(
        *_screen_profile_pick(
            profile,
            (0.003, 0.012),
            (0.008, 0.024),
            (0.014, 0.040),
        )
    )

    band = 1.0 + amplitude * np.sin(2.0 * np.pi * frequency * y + phase)

    return _to_image(array * band[:, None, None])


def _screen_surface_reflection_soft(image: Image.Image, profile: str) -> Image.Image:
    """Very weak glare only.

    The previous reflection helper could wash out large regions. This version
    should be rare and restrained.
    """
    array = _to_array(image)
    height, width = array.shape[:2]

    yy, xx = np.meshgrid(
        np.linspace(0.0, 1.0, height, dtype=np.float32),
        np.linspace(0.0, 1.0, width, dtype=np.float32),
        indexing="ij",
    )

    center_x = random.uniform(-0.20, 1.20)
    center_y = random.uniform(-0.20, 1.20)

    sigma_x = random.uniform(
        *_screen_profile_pick(
            profile,
            (0.25, 0.55),
            (0.20, 0.45),
            (0.16, 0.36),
        )
    )

    sigma_y = random.uniform(
        *_screen_profile_pick(
            profile,
            (0.12, 0.30),
            (0.10, 0.26),
            (0.08, 0.22),
        )
    )

    glare = np.exp(
        -(
            ((xx - center_x) ** 2) / (2.0 * sigma_x * sigma_x)
            + ((yy - center_y) ** 2) / (2.0 * sigma_y * sigma_y)
        )
    )[..., None]

    strength = random.uniform(
        *_screen_profile_pick(
            profile,
            (0.004, 0.014),
            (0.008, 0.028),
            (0.014, 0.050),
        )
    )

    tint = np.asarray(
        [
            random.uniform(0.96, 1.03),
            random.uniform(0.96, 1.03),
            random.uniform(0.94, 1.04),
        ],
        dtype=np.float32,
    )

    out = array * (1.0 - 0.04 * strength * glare) + strength * glare * tint

    return _to_image(out)


def _screen_photo_tone_map(image: Image.Image, profile: str) -> Image.Image:
    """Phone-camera-like tone response.

    Real screen photos usually keep strong blacks and may gain contrast. Avoid
    bloom/washout.
    """
    array = _to_array(image)

    exposure = random.uniform(
        *_screen_profile_pick(
            profile,
            (0.99, 1.04),
            (0.97, 1.06),
            (0.94, 1.08),
        )
    )

    contrast = random.uniform(
        *_screen_profile_pick(
            profile,
            (1.03, 1.10),
            (1.04, 1.14),
            (1.02, 1.16),
        )
    )

    gamma = random.uniform(
        *_screen_profile_pick(
            profile,
            (0.97, 1.03),
            (0.96, 1.05),
            (0.94, 1.07),
        )
    )

    # Very mild white balance drift.
    white_balance = np.asarray(
        [
            random.uniform(0.985, 1.025),
            random.uniform(0.985, 1.025),
            random.uniform(0.970, 1.030),
        ],
        dtype=np.float32,
    )

    out = np.clip(array * exposure, 0.0, 1.0)
    out = np.clip((out - 0.5) * contrast + 0.5, 0.0, 1.0)
    out = np.clip(out, 0.0, 1.0) ** gamma
    out = out * white_balance

    return _to_image(out)


def _screen_jpeg_output_profiled(image: Image.Image, profile: str) -> Image.Image:
    quality = random.randint(
        *_screen_profile_pick(
            profile,
            (95, 99),  # mild
            (88, 97),  # medium
            (72, 92),  # extra_hard
        )
    )

    return _save_reopen(image, "JPEG", quality=quality)


def _screen_photo_work_image(image: Image.Image, profile: str) -> Image.Image:
    """Keep a fairly large work image so screen texture survives resizing."""
    width, height = image.size
    long_edge = max(width, height)

    max_edge = random.randint(
        *_screen_profile_pick(
            profile,
            (768, 960),     # mild
            (896, 1120),    # medium
            (960, 1280),    # extra_hard
        )
    )

    if long_edge <= max_edge:
        return image.convert("RGB")

    scale = max_edge / float(long_edge)
    size = (
        max(8, int(round(width * scale))),
        max(8, int(round(height * scale))),
    )

    return image.convert("RGB").resize(size, Image.Resampling.BICUBIC)


def _screen_camera_resample(
    image: Image.Image,
    target_size: tuple[int, int],
    severity: str,
) -> Image.Image:
    """Sample the LCD image with a slightly mismatched camera sensor grid."""
    width, height = target_size

    scale = _uniform(severity, (0.94, 1.06), (0.88, 1.12), (0.78, 1.22))
    sensor_size = (
        max(8, int(round(width * scale))),
        max(8, int(round(height * scale))),
    )

    interpolation = random.choice(
        [
            Image.Resampling.BILINEAR,
            Image.Resampling.BICUBIC,
            Image.Resampling.LANCZOS,
        ]
    )

    out = image.resize(sensor_size, interpolation)
    return out.resize(target_size, interpolation)


def _bayer_cfa_capture_and_demosaic(image: Image.Image, severity: str) -> Image.Image:
    """Sample through a random-phase RGGB Bayer CFA, add noise, then demosaic."""
    array = _to_array(image)
    height, width = array.shape[:2]
    rng = _rng()

    y_phase = random.randint(0, 1)
    x_phase = random.randint(0, 1)

    y_even = ((np.arange(height) + y_phase) % 2) == 0
    x_even = ((np.arange(width) + x_phase) % 2) == 0

    yy = y_even[:, None]
    xx = x_even[None, :]

    red_mask = yy & xx
    blue_mask = (~yy) & (~xx)
    green_mask = ~(red_mask | blue_mask)

    raw = np.empty((height, width), dtype=np.float32)
    raw[red_mask] = array[..., 0][red_mask]
    raw[green_mask] = array[..., 1][green_mask]
    raw[blue_mask] = array[..., 2][blue_mask]

    # Sensor shot noise plus additive read noise.
    shot_scale = _pick(severity, 150.0, 90.0, 45.0)
    gaussian_sigma = _uniform(
        severity,
        (0.002, 0.010),
        (0.006, 0.020),
        (0.014, 0.040),
    )

    raw = rng.poisson(np.clip(raw, 0.0, 1.0) * shot_scale) / shot_scale
    raw += rng.normal(0.0, gaussian_sigma, size=raw.shape).astype(np.float32)
    raw = np.clip(raw, 0.0, 1.0)

    red = _interpolate_sparse_bayer_channel(raw, red_mask)
    green = _interpolate_sparse_bayer_channel(raw, green_mask)
    blue = _interpolate_sparse_bayer_channel(raw, blue_mask)

    return _to_image(np.stack([red, green, blue], axis=-1))


def _interpolate_sparse_bayer_channel(raw: np.ndarray, mask: np.ndarray) -> np.ndarray:
    kernel = np.array(
        [
            [1.0, 2.0, 1.0],
            [2.0, 4.0, 2.0],
            [1.0, 2.0, 1.0],
        ],
        dtype=np.float32,
    )

    values = np.where(mask, raw, 0.0).astype(np.float32)
    weights = mask.astype(np.float32)

    numerator = _convolve2d_same(values, kernel)
    denominator = _convolve2d_same(weights, kernel)

    interpolated = numerator / np.maximum(denominator, 1e-6)
    interpolated[mask] = raw[mask]

    return interpolated


def _convolve2d_same(array: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    array = array.astype(np.float32, copy=False)
    kernel = kernel.astype(np.float32, copy=False)

    height, width = array.shape
    kernel_height, kernel_width = kernel.shape

    pad_y = kernel_height // 2
    pad_x = kernel_width // 2

    padded = np.pad(array, ((pad_y, pad_y), (pad_x, pad_x)), mode="edge")
    out = np.zeros_like(array, dtype=np.float32)

    for y in range(kernel_height):
        for x in range(kernel_width):
            out += kernel[y, x] * padded[y : y + height, x : x + width]

    return out

def _screen_camera_resample_profiled(image: Image.Image, profile: str) -> Image.Image:
    width, height = image.size

    scale = random.uniform(
        *_screen_profile_pick(
            profile,
            (0.985, 1.015),
            (0.965, 1.040),
            (0.925, 1.080),
        )
    )

    if abs(scale - 1.0) < 0.004:
        return image

    sensor_size = (
        max(8, int(round(width * scale))),
        max(8, int(round(height * scale))),
    )

    interpolation = random.choice(
        [
            Image.Resampling.BILINEAR,
            Image.Resampling.BICUBIC,
            Image.Resampling.LANCZOS,
        ]
    )

    return image.resize(sensor_size, interpolation).resize((width, height), interpolation)

def _camera_screen_photo_with_profile(
    image: Image.Image,
    profile: str,
) -> Image.Image:
    """Camera-captured screen simulation tuned for full-frame training images.

    This version avoids the destructive full LCD/Bayer pipeline for normal
    samples. It models the most useful screen-photo artifacts directly:

    - mild perspective/camera sampling mismatch
    - fine vertical LCD texture
    - low-amplitude moiré
    - mild rolling-shutter banding
    - restrained sensor noise
    - camera-like tone and JPEG output

    The output remains readable for mild/medium profiles.
    """
    original_size = image.size
    pipeline_severity = _screen_photo_profile_to_severity(profile)

    work = _screen_photo_work_image(image, pipeline_severity)
    out = work.convert("RGB")

    # Small screen/camera perspective. Do not overuse this; the detector should
    # not learn that black borders imply "screen photo".
    if random.random() < _screen_profile_pick(profile, 0.08, 0.24, 0.55):
        out = _screen_projective_transform_soft(out, profile)

    # Slight camera sampling mismatch. This helps create real-looking aliasing
    # without turning the image into pure blur.
    if random.random() < _screen_profile_pick(profile, 0.35, 0.65, 0.90):
        out = _screen_camera_resample_profiled(out, profile)

    # Very mild optical blur. Your previous mild output looked too blurry, so
    # this must be weak.
    if random.random() < _screen_profile_pick(profile, 0.20, 0.40, 0.65):
        out = _screen_optical_blur_profiled(out, profile)

    # Rare, partial Bayer/CFA simulation. The full version is too destructive,
    # so blend it back into the pre-Bayer image.
    if random.random() < _screen_profile_pick(profile, 0.00, 0.025, 0.12):
        before_bayer = out
        bayer_severity = "train" if profile != "extra_hard" else "val"
        bayered = _bayer_cfa_capture_and_demosaic(out, bayer_severity)
        bayer_alpha = random.uniform(
            *_screen_profile_pick(
                profile,
                (0.00, 0.00),
                (0.08, 0.18),
                (0.14, 0.32),
            )
        )
        out = Image.blend(before_bayer, bayered, alpha=bayer_alpha)

    # Low sensor noise. Keep chroma noise restrained.
    out = _screen_capture_noise_profiled(out, profile)

    # Resize back before adding final screen texture, otherwise the texture gets
    # blurred away and mild samples look like ordinary blur.
    out = _resize_back(out, original_size)

    # Fine LCD/subpixel texture. This is the main visible screen-photo cue.
    out = _screen_fine_lcd_texture_overlay(out, profile)

    # Coarse moiré. Use often enough to be visible, but keep amplitude low.
    if random.random() < _screen_profile_pick(profile, 0.45, 0.68, 0.82):
        out = _screen_soft_coarse_moire(out, profile)

    # Rolling shutter / refresh mismatch. Subtle only.
    if random.random() < _screen_profile_pick(profile, 0.10, 0.22, 0.38):
        out = _screen_subtle_rolling_band(out, profile)

    # Reflections/glare caused the washed-out look before. Make them rare and weak.
    if random.random() < _screen_profile_pick(profile, 0.00, 0.035, 0.08):
        out = _screen_surface_reflection_soft(out, profile)

    # Real screen photos often have more contrast than the original, not less.
    out = _screen_photo_tone_map(out, profile)

    # Final camera/export compression.
    out = _screen_jpeg_output_profiled(out, profile)

    return out.convert("RGB")


OPERATIONS.update({
    "gaussian_blur": _gaussian_blur,
    "lens_blur": _lens_blur,
    "motion_blur": _motion_blur,
    "glass_blur": _glass_blur,
    "jpeg": _jpeg,
    "jpeg2000": _jpeg2000,
    "webp": _webp,
    "avif": _avif,
    "jxl": _jxl,
    "multi_jpeg": _multi_jpeg,
    "multi_codec": _multi_codec,
    "jpeg_then_jpeg2000": _jpeg_then_jpeg2000,
    "resize_then_codec": _resize_then_codec,
    "platform_recompress": _platform_recompress,
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
    "screenshot_resample": _screenshot_resample,
    "letterbox_pad": _letterbox_pad,
    "subtle_watermark": _subtle_watermark,
    "print_scan": _print_scan,
    "camera_screen_photo": _camera_screen_photo,
})