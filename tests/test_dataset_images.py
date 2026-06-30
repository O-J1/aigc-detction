from __future__ import annotations

import warnings

import pytest
from PIL import Image

from micv.data.dataset import AIGCManifestDataset, load_rgb_image


class RecordAwareTransform:
    needs_record = True

    def __init__(self) -> None:
        self.seen_path = None

    def __call__(self, image, record):
        self.seen_path = record.path
        return image


def test_manifest_dataset_skips_invalid_images_at_init(tmp_path) -> None:
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

    assert len(dataset) == 1
    assert dataset.skipped_bad_images == [invalid_path]
    assert dataset.deleted_bad_images == []
    assert dataset[0]["path"] == str(valid_path)


def test_manifest_dataset_deletes_invalid_images_when_enabled(tmp_path) -> None:
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

    dataset = AIGCManifestDataset(manifest_path, split="train", delete_invalid_images=True)

    assert len(dataset) == 1
    assert dataset.skipped_bad_images == [invalid_path]
    assert dataset.deleted_bad_images == [invalid_path]
    assert valid_path.exists()
    assert not invalid_path.exists()


def test_delete_invalid_images_requires_skip_policy(tmp_path) -> None:
    manifest_path = tmp_path / "manifest.csv"
    manifest_path.write_text("path,label,split\nmissing.png,0,train\n", encoding="utf-8")

    with pytest.raises(ValueError, match="delete_invalid_images requires"):
        AIGCManifestDataset(
            manifest_path,
            split="train",
            bad_image_policy="raise",
            delete_invalid_images=True,
        )


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
        "path,label,split\n"
        f"{image_path.name},0,val\n",
        encoding="utf-8",
    )
    transform = RecordAwareTransform()
    dataset = AIGCManifestDataset(manifest_path, split="val", transform=transform)

    _ = dataset[0]

    assert transform.seen_path == image_path
