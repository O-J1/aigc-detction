from __future__ import annotations

import csv
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset, WeightedRandomSampler

from micv.data.labels import FAKE_CLASS_NAMES, REAL_CLASS_NAMES

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class ManifestRecord:
    path: Path
    label: int
    split: str | None
    metadata: dict[str, Any]


class AIGCManifestDataset(Dataset[dict[str, Any]]):
    """Manifest-backed image dataset for real-vs-AI detection."""

    def __init__(
        self,
        manifest_path: str | Path,
        split: str | None = None,
        root_dir: str | Path | None = None,
        transform: Any | None = None,
        bad_image_policy: str = "skip",
        require_label: bool = True,
    ) -> None:
        if bad_image_policy not in {"raise", "skip", "zero"}:
            raise ValueError("bad_image_policy must be one of: raise, skip, zero")
        self.manifest_path = Path(manifest_path)
        self.root_dir = Path(root_dir) if root_dir is not None else self.manifest_path.parent
        self.transform = transform
        self.bad_image_policy = bad_image_policy
        self.require_label = require_label
        self.skipped_bad_images: list[Path] = []
        self.records = self._load_records(split=split)
        if self.bad_image_policy in {"raise", "skip"}:
            self.records = self._filter_loadable_records(self.records)
        if not self.records:
            raise ValueError(f"No records found in {self.manifest_path} for split={split!r}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        image = self._load_image(record.path)
        if self.transform is not None:
            image = self.transform(image)
        label = torch.tensor(record.label, dtype=torch.float32)
        return {
            "image": image,
            "label": label,
            "path": str(record.path),
            "metadata": record.metadata,
        }

    @property
    def labels(self) -> list[int]:
        return [record.label for record in self.records]

    def _load_records(self, split: str | None) -> list[ManifestRecord]:
        suffix = self.manifest_path.suffix.lower()
        if suffix == ".csv":
            rows = self._read_csv_rows()
        elif suffix == ".parquet":
            rows = self._read_parquet_rows()
        else:
            raise ValueError(f"Unsupported manifest format: {self.manifest_path.suffix}")

        records: list[ManifestRecord] = []
        for row in rows:
            row_split = row.get("split")
            if split is not None and row_split not in {None, ""} and row_split != split:
                continue
            if "path" not in row:
                raise ValueError("Manifest rows must include a path column.")
            if self.require_label and "label" not in row:
                raise ValueError("Training/evaluation manifests must include a label column.")
            image_path = self._resolve_path(str(row["path"]))
            label = normalize_label(row["label"]) if "label" in row and row["label"] != "" else -1
            metadata = {key: value for key, value in row.items() if key not in {"path", "label"}}
            records.append(ManifestRecord(path=image_path, label=label, split=row_split, metadata=metadata))
        return records

    def _read_csv_rows(self) -> list[dict[str, Any]]:
        with self.manifest_path.open("r", newline="", encoding="utf-8") as manifest_file:
            return list(csv.DictReader(manifest_file))

    def _read_parquet_rows(self) -> list[dict[str, Any]]:
        try:
            import pandas as pd
        except ImportError as error:
            raise ImportError("Install pandas and pyarrow to read Parquet manifests.") from error
        dataframe = pd.read_parquet(self.manifest_path)
        return dataframe.to_dict(orient="records")

    def _resolve_path(self, path_value: str) -> Path:
        image_path = Path(path_value)
        if image_path.is_absolute():
            return image_path
        return self.root_dir / image_path

    def _filter_loadable_records(self, records: list[ManifestRecord]) -> list[ManifestRecord]:
        filtered_records: list[ManifestRecord] = []
        for record in records:
            try:
                verify_image(record.path)
            except Exception as error:
                if self.bad_image_policy == "skip":
                    self.skipped_bad_images.append(record.path)
                    continue
                raise ValueError(f"Invalid image in {self.manifest_path}: {record.path}") from error
            filtered_records.append(record)
        return filtered_records

    def _load_image(self, image_path: Path) -> Image.Image:
        try:
            return load_rgb_image(image_path)
        except Exception:
            if self.bad_image_policy == "zero":
                return Image.new("RGB", (512, 512), color=(0, 0, 0))
            raise


def verify_image(image_path: str | Path) -> None:
    with Image.open(image_path) as image:
        image.verify()


def load_rgb_image(
    image_path: str | Path,
    background_color: tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    with Image.open(image_path) as image:
        image.load()
        if image.mode == "P" and "transparency" in image.info:
            image = image.convert("RGBA")
        if "A" in image.getbands():
            rgba_image = image.convert("RGBA")
            background = Image.new("RGBA", rgba_image.size, (*background_color, 255))
            background.alpha_composite(rgba_image)
            return background.convert("RGB")
        return image.convert("RGB")


def normalize_label(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(value, float) and math.isnan(value):
            raise ValueError("Label value cannot be NaN.")
        numeric_label = int(value)
        if numeric_label in {0, 1} and float(value) == float(numeric_label):
            return numeric_label
    normalized = str(value).strip().lower()
    if normalized in {"0", "real", "human", "natural", "negative"}:
        return 0
    if normalized in {"1", "fake", "ai", "aigc", "generated", "synthetic", "positive"}:
        return 1
    raise ValueError(f"Unsupported label value: {value!r}")


def make_weighted_sampler(dataset: AIGCManifestDataset) -> WeightedRandomSampler:
    label_counts = Counter(dataset.labels)
    sample_weights = [1.0 / label_counts[label] for label in dataset.labels]
    weights_tensor = torch.as_tensor(sample_weights, dtype=torch.double)
    return WeightedRandomSampler(weights_tensor, num_samples=len(weights_tensor), replacement=True)