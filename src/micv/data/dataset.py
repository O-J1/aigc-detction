from __future__ import annotations

import csv
import math
import random
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as distributed
from PIL import Image, ImageFile
from torch.nn import functional as torch_functional
from torch.utils.data import Dataset, Sampler, WeightedRandomSampler
from torch.utils.data.dataloader import default_collate

ImageFile.LOAD_TRUNCATED_IMAGES = True


class MultiResolutionBatchCollate:
    """Collate that resizes each batch to a randomly sampled square resolution.

    The dataset transform emits a fixed base resolution; resampling at batch
    level keeps token counts matched across committee slots within a batch.
    Randomness comes from the ``random`` module so PyTorch's per-worker and
    per-rank seeding applies.
    """

    def __init__(self, image_sizes: Sequence[int]) -> None:
        sizes = sorted({int(size) for size in image_sizes})
        if not sizes:
            raise ValueError("image_sizes must not be empty")
        self.image_sizes = sizes

    def __call__(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        batch = default_collate(samples)
        size = random.choice(self.image_sizes)
        images = batch["image"]
        if images.shape[-2:] != (size, size):
            flattened = images.reshape(-1, *images.shape[-3:]) if images.ndim == 5 else images
            resized = torch_functional.interpolate(
                flattened,
                size=(size, size),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )
            if images.ndim == 5:
                resized = resized.reshape(*images.shape[:2], *resized.shape[-3:])
            batch["image"] = resized
        return batch


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
        bad_image_policy: str = "raise",
        require_label: bool = True,
    ) -> None:
        if bad_image_policy not in {"raise", "zero"}:
            raise ValueError("bad_image_policy must be one of: raise, zero")
        self.manifest_path = Path(manifest_path)
        self.root_dir = Path(root_dir) if root_dir is not None else self.manifest_path.parent
        self.transform = transform
        self.bad_image_policy = bad_image_policy
        self.require_label = require_label
        self.records = self._load_records(split=split)
        if not self.records:
            raise ValueError(f"No records found in {self.manifest_path} for split={split!r}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        image = self._load_image(record.path)
        if self.transform is not None:
            if getattr(self.transform, "needs_record", False):
                image = self.transform(image, record)
            else:
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
        for row_number, row in enumerate(rows, start=2):
            row_split = _normalize_split_value(row.get("split"))
            if split is not None:
                if row_split is None:
                    manifest_path = row.get("path", "<missing path>")
                    raise ValueError(
                        f"Manifest row {row_number} in {self.manifest_path} is missing split "
                        f"for requested split={split!r}: {manifest_path}"
                    )
                if row_split != split:
                    continue
            if "path" not in row:
                raise ValueError("Manifest rows must include a path column.")
            if self.require_label and "label" not in row:
                raise ValueError("Training/evaluation manifests must include a label column.")
            manifest_path = str(row["path"])
            image_path = self._resolve_path(manifest_path)
            label = normalize_label(row["label"]) if "label" in row and row["label"] != "" else -1
            metadata = {key: value for key, value in row.items() if key not in {"path", "label"}}
            metadata.setdefault("manifest_path", manifest_path)
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

    def _load_image(self, image_path: Path) -> Image.Image:
        try:
            return load_rgb_image(image_path)
        except Exception:
            if self.bad_image_policy == "zero":
                return Image.new("RGB", (512, 512), color=(0, 0, 0))
            raise


def verify_image(image_path: str | Path) -> None:
    load_rgb_image(image_path)


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


def _normalize_split_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    split = str(value).strip()
    return split or None


class DistributedWeightedSampler(Sampler[int]):
    """Weighted sampler that draws globally, then shards samples by distributed rank."""

    def __init__(
        self,
        dataset: AIGCManifestDataset,
        num_replicas: int | None = None,
        rank: int | None = None,
        replacement: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        if num_replicas is None:
            if not distributed.is_available() or not distributed.is_initialized():
                raise RuntimeError("Requires distributed package to be available and initialized")
            num_replicas = distributed.get_world_size()
        if rank is None:
            if not distributed.is_available() or not distributed.is_initialized():
                raise RuntimeError("Requires distributed package to be available and initialized")
            rank = distributed.get_rank()
        if rank < 0 or rank >= num_replicas:
            raise ValueError(f"Invalid rank {rank}; rank must be in [0, {num_replicas - 1}]")

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.replacement = replacement
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0
        self.weights = _make_label_weights(dataset)

        if self.drop_last and len(self.dataset) % self.num_replicas != 0:
            self.num_samples = math.ceil((len(self.dataset) - self.num_replicas) / self.num_replicas)
        else:
            self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas
        if not self.replacement and self.total_size > len(self.dataset):
            raise ValueError("replacement=False requires drop_last=True when dataset is not evenly divisible")

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        global_indices = torch.multinomial(
            self.weights,
            self.total_size,
            self.replacement,
            generator=generator,
        ).tolist()
        rank_indices = global_indices[self.rank : self.total_size : self.num_replicas]
        return iter(rank_indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


def _make_label_weights(dataset: AIGCManifestDataset) -> torch.Tensor:
    label_counts = Counter(dataset.labels)
    sample_weights = [1.0 / label_counts[label] for label in dataset.labels]
    return torch.as_tensor(sample_weights, dtype=torch.double)


def make_weighted_sampler(dataset: AIGCManifestDataset) -> WeightedRandomSampler:
    weights_tensor = _make_label_weights(dataset)
    return WeightedRandomSampler(weights_tensor, num_samples=len(weights_tensor), replacement=True)