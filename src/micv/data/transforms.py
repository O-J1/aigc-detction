from __future__ import annotations

import io
import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import Sequence

import torch
from PIL import Image, ImageFilter
from torch import Tensor
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as transform_functional

DEFAULT_MEAN = (0.485, 0.456, 0.406)
DEFAULT_STD = (0.229, 0.224, 0.225)


class RandomJPEGCompression:
    def __init__(self, quality_range: tuple[int, int] = (35, 95), probability: float = 0.5) -> None:
        self.quality_range = quality_range
        self.probability = probability

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() > self.probability:
            return image
        quality = random.randint(*self.quality_range)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        with Image.open(buffer) as compressed:
            compressed.load()
            return compressed.convert("RGB")


class RandomGaussianNoise:
    def __init__(self, sigma_range: tuple[float, float] = (0.005, 0.05), probability: float = 0.25) -> None:
        self.sigma_range = sigma_range
        self.probability = probability

    def __call__(self, image_tensor: Tensor) -> Tensor:
        if random.random() > self.probability:
            return image_tensor
        sigma = random.uniform(*self.sigma_range)
        noise = torch.randn_like(image_tensor) * sigma
        return torch.clamp(image_tensor + noise, min=0.0, max=1.0)


@dataclass
class HierarchicalAugmentation:
    difficulty: str = "mixed"

    def __call__(self, image: Image.Image) -> Image.Image:
        if self.difficulty in {"none", "off", "disabled"}:
            return image
        level = self._choose_level()
        num_steps = {"easy": 1, "medium": 2, "hard": 3}[level]
        augmented = image
        for _ in range(num_steps):
            augmented = random.choice(self._operations(level))(augmented)
        return augmented

    def _choose_level(self) -> str:
        if self.difficulty in {"easy", "medium", "hard"}:
            return self.difficulty
        return random.choices(["easy", "medium", "hard"], weights=[0.5, 0.35, 0.15], k=1)[0]

    def _operations(self, level: str) -> list[Callable[[Image.Image], Image.Image]]:
        if level == "easy":
            return [self._blur_easy, self._jpeg_easy, self._affine_easy]
        if level == "medium":
            return [self._blur_medium, self._jpeg_medium, self._affine_medium, self._resize_artifact]
        return [self._blur_hard, self._jpeg_hard, self._affine_hard, self._perspective]

    @staticmethod
    def _blur_easy(image: Image.Image) -> Image.Image:
        return image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 1.0)))

    @staticmethod
    def _blur_medium(image: Image.Image) -> Image.Image:
        return image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.8, 1.8)))

    @staticmethod
    def _blur_hard(image: Image.Image) -> Image.Image:
        return image.filter(ImageFilter.GaussianBlur(radius=random.uniform(1.5, 3.0)))

    @staticmethod
    def _jpeg_easy(image: Image.Image) -> Image.Image:
        return RandomJPEGCompression((65, 95), probability=1.0)(image)

    @staticmethod
    def _jpeg_medium(image: Image.Image) -> Image.Image:
        return RandomJPEGCompression((40, 85), probability=1.0)(image)

    @staticmethod
    def _jpeg_hard(image: Image.Image) -> Image.Image:
        return RandomJPEGCompression((25, 70), probability=1.0)(image)

    @staticmethod
    def _affine_easy(image: Image.Image) -> Image.Image:
        return _random_affine(image, max_degrees=5.0, max_translate=0.02, min_scale=0.98, max_scale=1.02)

    @staticmethod
    def _affine_medium(image: Image.Image) -> Image.Image:
        return _random_affine(image, max_degrees=10.0, max_translate=0.05, min_scale=0.95, max_scale=1.05)

    @staticmethod
    def _affine_hard(image: Image.Image) -> Image.Image:
        return _random_affine(image, max_degrees=15.0, max_translate=0.08, min_scale=0.9, max_scale=1.1)

    @staticmethod
    def _resize_artifact(image: Image.Image) -> Image.Image:
        width, height = image.size
        scale = random.uniform(0.45, 0.8)
        resized = image.resize((max(1, int(width * scale)), max(1, int(height * scale))), Image.Resampling.BICUBIC)
        return resized.resize((width, height), Image.Resampling.BICUBIC)

    @staticmethod
    def _perspective(image: Image.Image) -> Image.Image:
        return transforms.RandomPerspective(distortion_scale=0.25, p=1.0, fill=0)(image)


class StaticValidationDegradation:
    """Deterministic-ish validation degradation profile for robustness reporting."""

    def __call__(self, image: Image.Image) -> Image.Image:
        degraded = image.filter(ImageFilter.GaussianBlur(radius=0.7))
        return RandomJPEGCompression((75, 75), probability=1.0)(degraded)


def build_train_transform(
    image_size: int = 512,
    difficulty: str = "mixed",
    mean: Sequence[float] = DEFAULT_MEAN,
    std: Sequence[float] = DEFAULT_STD,
) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(
                image_size,
                scale=(0.65, 1.0),
                ratio=(0.9, 1.1),
                interpolation=InterpolationMode.BICUBIC,
            ),
            HierarchicalAugmentation(difficulty=difficulty),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            RandomGaussianNoise(probability=0.25 if difficulty != "none" else 0.0),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def build_eval_transform(
    image_size: int = 512,
    mean: Sequence[float] = DEFAULT_MEAN,
    std: Sequence[float] = DEFAULT_STD,
    static_augmentation: bool = False,
) -> transforms.Compose:
    transform_steps: list[object] = [
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
    ]
    if static_augmentation:
        transform_steps.append(StaticValidationDegradation())
    transform_steps.extend([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    return transforms.Compose(transform_steps)


def _random_affine(
    image: Image.Image,
    max_degrees: float,
    max_translate: float,
    min_scale: float,
    max_scale: float,
) -> Image.Image:
    width, height = image.size
    angle = random.uniform(-max_degrees, max_degrees)
    translate = (
        int(random.uniform(-max_translate, max_translate) * width),
        int(random.uniform(-max_translate, max_translate) * height),
    )
    scale = random.uniform(min_scale, max_scale)
    return transform_functional.affine(
        image,
        angle=angle,
        translate=translate,
        scale=scale,
        shear=[0.0, 0.0],
        interpolation=InterpolationMode.BICUBIC,
        fill=0,
    )