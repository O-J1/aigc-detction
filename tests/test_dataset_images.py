from __future__ import annotations

import warnings

from PIL import Image

from micv.data.dataset import AIGCManifestDataset, load_rgb_image


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
    assert dataset[0]["path"] == str(valid_path)


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
