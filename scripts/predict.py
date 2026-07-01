from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from micv.data import (
    AIGCManifestDataset,
    FAKE_CLASS_NAMES,
    REAL_CLASS_NAMES,
    build_eval_transform,
    load_rgb_image,
    verify_image,
)
from micv.models import MICVDualStreamEnsemble
from micv.training.trainer import load_checkpoint
from micv.utils import load_config

IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


class ImagePathDataset(Dataset):
    def __init__(self, paths: list[Path], transform, labels: dict[Path, int] | None = None) -> None:
        self.paths = paths
        self.transform = transform
        self.labels = labels or {}

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        image_path = self.paths[index]
        image_tensor = self.transform(load_rgb_image(image_path))
        return {
            "image": image_tensor,
            "path": str(image_path),
            "label": self.labels.get(image_path, -1),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MICV batch prediction.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--input",
        required=True,
        help="Image file, image directory, or manifest path.",
    )
    parser.add_argument("--output", required=True, help="Output CSV path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eval_transform = build_eval_transform(
        image_size=config.data.image_size,
        mean=config.augmentation.mean,
        std=config.augmentation.std,
        static_augmentation=False,
    )
    input_path = Path(args.input)
    if input_path.suffix.lower() in {".csv", ".parquet"}:
        dataset = AIGCManifestDataset(
            input_path,
            split=None,
            root_dir=config.data.root_dir,
            transform=eval_transform,
            bad_image_policy=config.data.bad_image_policy,
            require_label=False,
        )
        has_ground_truth = _has_manifest_ground_truth(dataset)
    else:
        paths = _collect_image_paths(input_path)
        labels = _infer_directory_ground_truth(input_path, paths) if input_path.is_dir() else {}
        has_ground_truth = _has_binary_ground_truth(labels.values())
        dataset = ImagePathDataset(
            paths,
            eval_transform,
            labels=labels if has_ground_truth else None,
        )

    data_loader = DataLoader(
        dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
    )
    model = MICVDualStreamEnsemble.from_config(asdict(config.model)).to(device)
    load_checkpoint(args.checkpoint, model=model, map_location=device)
    model.eval()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        fieldnames = ["path", "prob_ai", "prediction"]
        if has_ground_truth:
            fieldnames.extend(["ground_truth", "correct"])
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        correct_predictions = 0
        labeled_predictions = 0
        with torch.no_grad():
            for batch in data_loader:
                images = batch["image"].to(device)
                outputs = model(images)
                probabilities = outputs["fused_prob"].detach().cpu().flatten().tolist()
                labels = batch.get("label", [-1] * len(probabilities))
                rows = zip(batch["path"], probabilities, labels, strict=True)
                for image_path, probability, label in rows:
                    prediction = int(probability >= 0.5)
                    row = {
                        "path": image_path,
                        "prob_ai": f"{probability:.8f}",
                        "prediction": prediction,
                    }
                    if has_ground_truth:
                        ground_truth = _as_int_label(label)
                        correct = ""
                        if ground_truth in {0, 1}:
                            correct = int(prediction == ground_truth)
                            correct_predictions += correct
                            labeled_predictions += 1
                        row.update({"ground_truth": ground_truth, "correct": correct})
                    writer.writerow(row)

    if has_ground_truth:
        accuracy = correct_predictions / max(1, labeled_predictions)
        print(
            f"accuracy={accuracy:.4f} "
            f"correct={correct_predictions} total={labeled_predictions}"
        )


def _collect_image_paths(input_path: Path) -> list[Path]:
    if input_path.is_file():
        paths = [input_path]
    else:
        paths = sorted(
            path for path in input_path.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES
        )
    valid_paths = []
    for path in paths:
        try:
            verify_image(path)
        except Exception:
            continue
        valid_paths.append(path)
    return valid_paths


def _infer_directory_ground_truth(root: Path, image_paths: list[Path]) -> dict[Path, int]:
    labels: dict[Path, int] = {}
    for image_path in image_paths:
        label = _label_from_relative_parts(image_path.relative_to(root).parts[:-1])
        if label is not None:
            labels[image_path] = label
    return labels


def _label_from_relative_parts(parts: tuple[str, ...]) -> int | None:
    real_names = {name.lower() for name in REAL_CLASS_NAMES}
    fake_names = {name.lower() for name in FAKE_CLASS_NAMES}
    for part in reversed(parts):
        normalized = part.lower()
        if normalized in real_names:
            return 0
        if normalized in fake_names:
            return 1
    return None


def _has_binary_ground_truth(labels) -> bool:
    label_set = set(labels)
    return 0 in label_set and 1 in label_set


def _has_manifest_ground_truth(dataset: AIGCManifestDataset) -> bool:
    return _has_binary_ground_truth(label for label in dataset.labels if label in {0, 1})


def _as_int_label(label) -> int:
    if torch.is_tensor(label):
        return int(label.item())
    return int(label)


if __name__ == "__main__":
    main()