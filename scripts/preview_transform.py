# scripts/preview_screen_photo_op.py

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from micv.data.transforms import apply_op


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def add_label(image: Image.Image, label: str) -> Image.Image:
    image = image.convert("RGB")
    label_height = 34

    canvas = Image.new(
        "RGB",
        (image.width, image.height + label_height),
        color=(255, 255, 255),
    )
    canvas.paste(image, (0, label_height))

    draw = ImageDraw.Draw(canvas)
    draw.text((10, 8), label, fill=(0, 0, 0))

    return canvas


def make_grid(images: list[tuple[str, Image.Image]]) -> Image.Image:
    labeled = [add_label(image, label) for label, image in images]

    width = max(image.width for image in labeled)
    height = max(image.height for image in labeled)

    padded = []
    for image in labeled:
        canvas = Image.new("RGB", (width, height), color=(255, 255, 255))
        canvas.paste(image, (0, 0))
        padded.append(canvas)

    grid = Image.new("RGB", (width * len(padded), height), color=(255, 255, 255))

    for index, image in enumerate(padded):
        grid.paste(image, (index * width, 0))

    return grid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/preview_screen_photo"),
    )
    parser.add_argument(
        "--op",
        type=str,
        default="camera_screen_photo",
    )
    parser.add_argument(
        "--severity",
        type=str,
        default="hard",
        choices=["train", "val", "hard", "mixed", "test"],
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--samples", type=int, default=4)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    image = Image.open(args.input).convert("RGB")

    outputs: list[tuple[str, Image.Image]] = [("original", image)]

    for sample_idx in range(args.samples):
        seed_everything(args.seed + sample_idx)

        transformed = apply_op(
            args.op,
            image,
            severity=args.severity,
        ).convert("RGB")

        output_path = (
            args.output_dir
            / f"{args.input.stem}_{args.op}_{args.severity}_{sample_idx:02d}.png"
        )

        # Save as PNG to avoid adding another compression layer after the op.
        transformed.save(output_path)
        outputs.append((f"{args.op} {args.severity} #{sample_idx}", transformed))

        print(f"saved: {output_path}")

    grid = make_grid(outputs)
    grid_path = args.output_dir / f"{args.input.stem}_{args.op}_{args.severity}_grid.png"
    grid.save(grid_path)

    print(f"saved grid: {grid_path}")


if __name__ == "__main__":
    main()