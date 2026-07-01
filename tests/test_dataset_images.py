from __future__ import annotations

import warnings

import pytest
from PIL import Image

from micv.data.dataset import AIGCManifestDataset, load_rgb_image


class RecordAwareTransform:
    needs_record = True

    def __init__(self) -> None:
        self.seen_path = None
        self.seen_metadata = None

    def __call__(self, image, record):
        self.seen_path = record.path
        self.seen_metadata = record.metadata
        return image


def test_manifest_dataset_does_not_preflight_invalid_images_at_init(tmp_path) -> None:
    valid_path = tmp_path / "valid.png"
    invalid_path = tmp_path / "invalid.png"
    manifest_path = tmp_path / "manifest.csv"

    Image.new("RGB", (4, 4), color=(255, 0, 0)).save(valid_path)
    invalid_path.write_bytes(b"not a png")
    manifest_path.write_text(
        "path,label,split\n"
        f"{valid_path.name},0,train\n"
        f"{invalid_path.name},1,train\n",
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
    assert not any("Palette images with Transparency" in str(warning.message) for warning in caught_warnings)


def test_manifest_dataset_passes_record_to_record_aware_transform(tmp_path) -> None:
    image_path = tmp_path / "image.png"
    manifest_path = tmp_path / "manifest.csv"
    Image.new("RGB", (4, 4), color=(0, 255, 0)).save(image_path)
    manifest_path.write_text(
        "path,label,split,md5\n"
        f"{image_path.name},0,val,abc123\n",
        encoding="utf-8",
    )
    transform = RecordAwareTransform()
    dataset = AIGCManifestDataset(manifest_path, split="val", transform=transform)

    _ = dataset[0]

    assert transform.seen_path == image_path
    assert transform.seen_metadata == {"split": "val", "md5": "abc123", "manifest_path": image_path.name}


def test_manifest_dataset_excludes_other_named_splits(tmp_path) -> None:
    train_path = tmp_path / "train.png"
    val_path = tmp_path / "val.png"
    manifest_path = tmp_path / "manifest.csv"
    Image.new("RGB", (4, 4), color=(255, 0, 0)).save(train_path)
    Image.new("RGB", (4, 4), color=(0, 255, 0)).save(val_path)
    manifest_path.write_text(
        "path,label,split\n"
        f"{train_path.name},0,train\n"
        f"{val_path.name},1,val\n",
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
        "path,label,split\n"
        f"{image_path.name},0,\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing split"):
        AIGCManifestDataset(manifest_path, split="train")


def test_manifest_dataset_rejects_missing_split_column_when_split_requested(tmp_path) -> None:
    image_path = tmp_path / "image.png"
    manifest_path = tmp_path / "manifest.csv"
    Image.new("RGB", (4, 4), color=(255, 0, 0)).save(image_path)
    manifest_path.write_text(
        "path,label\n"
        f"{image_path.name},0\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing split"):
        AIGCManifestDataset(manifest_path, split="train")
