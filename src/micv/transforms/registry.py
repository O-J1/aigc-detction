from __future__ import annotations

from collections.abc import Callable

from PIL import Image

from micv.transforms.severity import sample_effective_severity


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
        "webp",
        "avif",
        "jxl",
        "multi_jpeg",
        "multi_codec",
        "jpeg_then_jpeg2000",
        "resize_then_codec",
        "platform_recompress",
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
        "letterbox_pad",
    ],
    "capture": [
        "screenshot_resample",
        "subtle_watermark",
        "print_scan",
        "camera_screen_photo",
    ],
}

TRAIN_OPS = {
    "gaussian_blur",
    "lens_blur",
    "color_shift",
    "color_saturation",
    "jpeg",
    "webp",
    "avif",
    "resize_then_codec",
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
    "jpeg2000",
    "jxl",
    "multi_jpeg",
    "multi_codec",
    "jpeg_then_jpeg2000",
    "platform_recompress",
    "screenshot_resample",
    "letterbox_pad",
    "subtle_watermark",
    "camera_screen_photo",
}

Operation = Callable[[Image.Image, str], Image.Image]

OPERATIONS: dict[str, Operation] = {}


def apply_op(op_name: str, image: Image.Image, severity: str = "mixed") -> Image.Image:
    effective_severity = sample_effective_severity(severity)
    try:
        operation = OPERATIONS[op_name]
    except KeyError as error:
        raise ValueError(f"Unknown augmentation operation: {op_name}") from error
    return operation(image.convert("RGB"), effective_severity)