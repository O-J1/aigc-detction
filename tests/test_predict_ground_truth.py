from __future__ import annotations

from scripts.predict import (
    _has_binary_ground_truth,
    _infer_directory_ground_truth,
    _label_from_relative_parts,
)


def test_infer_directory_ground_truth_from_class_folders(tmp_path) -> None:
    real_image = tmp_path / "real" / "photo.jpg"
    fake_image = tmp_path / "generated" / "sample.jpg"
    unlabeled_image = tmp_path / "misc" / "unknown.jpg"

    labels = _infer_directory_ground_truth(tmp_path, [real_image, fake_image, unlabeled_image])

    assert labels == {real_image: 0, fake_image: 1}
    assert _has_binary_ground_truth(labels.values())


def test_ground_truth_requires_real_and_fake_labels(tmp_path) -> None:
    labels = _infer_directory_ground_truth(
        tmp_path,
        [
            tmp_path / "real" / "first.jpg",
            tmp_path / "authentic" / "second.jpg",
        ],
    )

    assert set(labels.values()) == {0}
    assert not _has_binary_ground_truth(labels.values())


def test_label_from_relative_parts_uses_nearest_class_folder() -> None:
    assert _label_from_relative_parts(("val", "real")) == 0
    assert _label_from_relative_parts(("val", "fake")) == 1
    assert _label_from_relative_parts(("fake", "real")) == 0
    assert _label_from_relative_parts(("unlabeled",)) is None