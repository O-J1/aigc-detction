from __future__ import annotations

import warnings

import pytest
import torch
from PIL import Image

from micv.data.dataset import AIGCManifestDataset, MultiResolutionBatchCollate, load_rgb_image


class RecordAwareTransform:
    needs_record = True

    def __init__(self) -> None:
        self.seen_path = None
        self.seen_metadata = None

    def __call__(self, image, record):
        self.seen_path = record.path
        self.seen_metadata = record.metadata
        return image


def test_multi_resolution_collate_resizes_batch_to_sampled_size() -> None:
    collate = MultiResolutionBatchCollate([32])
    samples = [
        {"image": torch.randn(3, 64, 64), "label": torch.tensor(1.0)},
        {"image": torch.randn(3, 64, 64), "label": torch.tensor(0.0)},
    ]

    batch = collate(samples)

    assert batch["image"].shape == (2, 3, 32, 32)
    assert batch["label"].shape == (2,)


def test_multi_resolution_collate_handles_multi_view_samples() -> None:
    collate = MultiResolutionBatchCollate([32])
    samples = [
        {"image": torch.randn(4, 3, 64, 64), "label": torch.tensor(1.0)},
        {"image": torch.randn(4, 3, 64, 64), "label": torch.tensor(0.0)},
    ]

    batch = collate(samples)

    assert batch["image"].shape == (2, 4, 3, 32, 32)


def test_multi_resolution_collate_rejects_empty_sizes() -> None:
    with pytest.raises(ValueError):
        MultiResolutionBatchCollate([])


def test_manifest_dataset_does_not_preflight_invalid_images_at_init(tmp_path) -> None:
    valid_path = tmp_path / "valid.png"
    invalid_path = tmp_path / "invalid.png"
    manifest_path = tmp_path / "manifest.csv"

    Image.new("RGB", (4, 4), color=(255, 0, 0)).save(valid_path)
    invalid_path.write_bytes(b"not a png")
    manifest_path.write_text(
        f"path,label,split\n{valid_path.name},0,train\n{invalid_path.name},1,train\n",
        encoding="utf-8",
    )

    dataset = AIGCManifestDataset(manifest_path, split="train")

    assert len(dataset) == 2
    assert dataset[0]["path"] == str(valid_path)
    with pytest.raises(Exception):
        _ = dataset[1]


def test_palette_transparency_image_loads_without_warning(tmp_path) -> None:
    image_path = tmp_path / "palette_transparency.png"
    image = Image.new("P", (2, 2))
    image.putpalette([0, 0, 0, 255, 0, 0] + [0, 0, 0] * 254)
    image.info["transparency"] = bytes([0, 255] + [255] * 254)
    image.save(image_path)

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        loaded = load_rgb_image(image_path)

    assert loaded.mode == "RGB"
    assert not any(
        "Palette images with Transparency" in str(warning.message) for warning in caught_warnings
    )


def test_manifest_dataset_passes_record_to_record_aware_transform(tmp_path) -> None:
    image_path = tmp_path / "image.png"
    manifest_path = tmp_path / "manifest.csv"
    Image.new("RGB", (4, 4), color=(0, 255, 0)).save(image_path)
    manifest_path.write_text(
        f"path,label,split,md5\n{image_path.name},0,val,abc123\n",
        encoding="utf-8",
    )
    transform = RecordAwareTransform()
    dataset = AIGCManifestDataset(manifest_path, split="val", transform=transform)

    _ = dataset[0]

    assert transform.seen_path == image_path
    assert transform.seen_metadata == {
        "split": "val",
        "md5": "abc123",
        "manifest_path": image_path.name,
    }


def test_manifest_dataset_excludes_other_named_splits(tmp_path) -> None:
    train_path = tmp_path / "train.png"
    val_path = tmp_path / "val.png"
    manifest_path = tmp_path / "manifest.csv"
    Image.new("RGB", (4, 4), color=(255, 0, 0)).save(train_path)
    Image.new("RGB", (4, 4), color=(0, 255, 0)).save(val_path)
    manifest_path.write_text(
        f"path,label,split\n{train_path.name},0,train\n{val_path.name},1,val\n",
        encoding="utf-8",
    )

    dataset = AIGCManifestDataset(manifest_path, split="val")

    assert len(dataset) == 1
    assert dataset.records[0].path == val_path


def test_manifest_dataset_rejects_blank_split_when_split_requested(tmp_path) -> None:
    image_path = tmp_path / "image.png"
    manifest_path = tmp_path / "manifest.csv"
    Image.new("RGB", (4, 4), color=(255, 0, 0)).save(image_path)
    manifest_path.write_text(
        f"path,label,split\n{image_path.name},0,\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing split"):
        AIGCManifestDataset(manifest_path, split="train")


def test_manifest_dataset_rejects_missing_split_column_when_split_requested(tmp_path) -> None:
    image_path = tmp_path / "image.png"
    manifest_path = tmp_path / "manifest.csv"
    Image.new("RGB", (4, 4), color=(255, 0, 0)).save(image_path)
    manifest_path.write_text(
        f"path,label\n{image_path.name},0\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing split"):
        AIGCManifestDataset(manifest_path, split="train")


def test_manifest_dataset_projects_metadata_columns(tmp_path) -> None:
    image_path = tmp_path / "image.png"
    manifest_path = tmp_path / "manifest.csv"
    Image.new("RGB", (4, 4), color=(0, 255, 0)).save(image_path)
    manifest_path.write_text(
        f"path,label,split,md5,unused\n{image_path.name},0,val,abc123,large-value\n",
        encoding="utf-8",
    )

    dataset = AIGCManifestDataset(
        manifest_path,
        split="val",
        metadata_columns=["md5"],
    )

    assert dataset.records[0].metadata == {
        "split": "val",
        "md5": "abc123",
        "manifest_path": image_path.name,
    }


def test_manifest_dataset_can_exclude_corrupt_images_with_statistics(tmp_path) -> None:
    valid_path = tmp_path / "valid.png"
    invalid_path = tmp_path / "invalid.png"
    manifest_path = tmp_path / "manifest.csv"
    Image.new("RGB", (4, 4), color=(255, 0, 0)).save(valid_path)
    invalid_path.write_bytes(b"not an image")
    manifest_path.write_text(
        f"path,label,split\n{valid_path.name},0,train\n{invalid_path.name},1,train\n",
        encoding="utf-8",
    )

    with pytest.warns(RuntimeWarning, match="Excluded 1 unreadable image"):
        dataset = AIGCManifestDataset(
            manifest_path,
            split="train",
            bad_image_policy="exclude",
        )

    assert len(dataset) == 1
    assert dataset.corruption_count == 1
    assert dataset.corrupt_paths == [str(invalid_path)]


def test_zero_bad_image_policy_is_visible_and_counted(tmp_path) -> None:
    invalid_path = tmp_path / "invalid.png"
    manifest_path = tmp_path / "manifest.csv"
    invalid_path.write_bytes(b"not an image")
    manifest_path.write_text(
        f"path,label,split\n{invalid_path.name},1,train\n",
        encoding="utf-8",
    )
    dataset = AIGCManifestDataset(
        manifest_path,
        split="train",
        bad_image_policy="zero",
    )

    with pytest.warns(RuntimeWarning, match="Replacing unreadable image"):
        sample = dataset[0]

    assert sample["image"].size == (512, 512)
    assert dataset.corruption_count == 1
    assert dataset.corrupt_paths == [str(invalid_path)]


def test_parquet_manifest_filters_split_and_projects_columns(tmp_path) -> None:
    pyarrow = pytest.importorskip("pyarrow")
    parquet = pytest.importorskip("pyarrow.parquet")
    image_path = tmp_path / "image.png"
    other_path = tmp_path / "other.png"
    manifest_path = tmp_path / "manifest.parquet"
    Image.new("RGB", (4, 4), color=(0, 255, 0)).save(image_path)
    Image.new("RGB", (4, 4), color=(255, 0, 0)).save(other_path)
    table = pyarrow.table(
        {
            "path": [image_path.name, other_path.name],
            "label": [0, 1],
            "split": ["val", "train"],
            "md5": ["abc123", "def456"],
            "unused": ["large-a", "large-b"],
        }
    )
    parquet.write_table(table, manifest_path)

    dataset = AIGCManifestDataset(
        manifest_path,
        split="val",
        metadata_columns=["md5"],
    )

    assert len(dataset) == 1
    assert dataset.records[0].path == image_path
    assert "unused" not in dataset.records[0].metadata
