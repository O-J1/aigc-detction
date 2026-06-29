from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BUILD_MANIFEST_PATH = PROJECT_ROOT / "scripts" / "build_manifest.py"
BUILD_MANIFEST_SPEC = importlib.util.spec_from_file_location("build_manifest", BUILD_MANIFEST_PATH)
assert BUILD_MANIFEST_SPEC is not None
build_manifest = importlib.util.module_from_spec(BUILD_MANIFEST_SPEC)
sys.modules[BUILD_MANIFEST_SPEC.name] = build_manifest
assert BUILD_MANIFEST_SPEC.loader is not None
BUILD_MANIFEST_SPEC.loader.exec_module(build_manifest)


def test_hybrid_manifest_preserves_forced_splits_and_random_split_is_deterministic(
    tmp_path,
    monkeypatch,
) -> None:
    train_real = _touch_image(tmp_path / "forced_train" / "real" / "train_real.jpg")
    val_fake = _touch_image(tmp_path / "forced_val" / "fake" / "val_fake.jpg")
    for index in range(4):
        _touch_image(tmp_path / "random_real" / f"real_{index}.jpg")
        _touch_image(tmp_path / "random_fake" / f"fake_{index}.jpg")

    spec_path = tmp_path / "hybrid.yaml"
    spec_path.write_text(
        f"""
root: {tmp_path.as_posix()}
seed: 123
random_train_fraction: 0.5
sources:
  - path: forced_train/real
    label: real
    split: train
    source_dataset: forced_train_real
  - path: forced_val/fake
    label: fake
    split: val
    source_dataset: forced_val_fake
    generator: heldout_generator
  - path: random_real
    label: real
    split: random
  - path: random_fake
    label: fake
    split: random
""",
        encoding="utf-8",
    )
    first_output = tmp_path / "manifest_first.csv"
    second_output = tmp_path / "manifest_second.csv"

    _run_builder(monkeypatch, spec_path, first_output)
    _run_builder(monkeypatch, spec_path, second_output)

    first_rows = _read_rows(first_output)
    second_rows = _read_rows(second_output)

    assert first_rows == second_rows
    assert len(first_rows) == 10
    assert _row_for_path(first_rows, train_real)["split"] == "train"
    val_fake_row = _row_for_path(first_rows, val_fake)
    assert val_fake_row["split"] == "val"
    assert val_fake_row["label"] == "1"
    assert val_fake_row["generator"] == "heldout_generator"

    random_rows = [row for row in first_rows if _normalized_path(row).startswith("random_")]
    assert {row["split"] for row in random_rows} == {"train", "val"}
    assert sum(row["split"] == "train" for row in random_rows) == 4
    assert sum(row["split"] == "val" for row in random_rows) == 4


def test_hybrid_manifest_can_assign_random_leaf_folders_as_units(tmp_path, monkeypatch) -> None:
    for folder_name in ["leaf_a", "leaf_b", "leaf_c", "leaf_d"]:
        for index in range(2):
            _touch_image(tmp_path / "random_real" / folder_name / f"image_{index}.jpg")

    spec_path = tmp_path / "hybrid_leaf.yaml"
    spec_path.write_text(
        f"""
root: {tmp_path.as_posix()}
seed: 7
random_train_fraction: 0.5
random_split_unit: leaf-folder
sources:
  - path: random_real
    label: real
    split: random
""",
        encoding="utf-8",
    )
    output_path = tmp_path / "manifest.csv"

    _run_builder(monkeypatch, spec_path, output_path)

    rows = _read_rows(output_path)
    splits_by_leaf: dict[str, set[str]] = {}
    for row in rows:
        leaf_name = _normalized_path(row).split("/")[-2]
        splits_by_leaf.setdefault(leaf_name, set()).add(row["split"])

    assert all(len(splits) == 1 for splits in splits_by_leaf.values())
    assert {next(iter(splits)) for splits in splits_by_leaf.values()} == {"train", "val"}


def test_hybrid_manifest_samples_within_each_leaf_folder(tmp_path, monkeypatch) -> None:
    for folder_name in ["leaf_a", "leaf_b"]:
        for index in range(4):
            _touch_image(tmp_path / "sampled" / folder_name / f"image_{index}.jpg")

    spec_path = tmp_path / "hybrid_sampled.yaml"
    spec_path.write_text(
        f"""
root: {tmp_path.as_posix()}
seed: 5
sources:
  - path: sampled
    label: real
    split: train
    keep_percent: 25
    min_per_leaf_folder: 1
""",
        encoding="utf-8",
    )
    output_path = tmp_path / "manifest.csv"

    _run_builder(monkeypatch, spec_path, output_path)

    rows = _read_rows(output_path)

    assert len(rows) == 2
    assert {_normalized_path(row).split("/")[-2] for row in rows} == {"leaf_a", "leaf_b"}
    assert {row["split"] for row in rows} == {"train"}


def test_hybrid_manifest_rejects_duplicate_paths_across_splits(tmp_path, monkeypatch) -> None:
    _touch_image(tmp_path / "shared" / "image.jpg")
    spec_path = tmp_path / "hybrid_duplicate.yaml"
    spec_path.write_text(
        f"""
root: {tmp_path.as_posix()}
sources:
  - path: shared
    label: real
    split: train
  - path: shared
    label: real
    split: val
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Duplicate image path"):
        _run_builder(monkeypatch, spec_path, tmp_path / "manifest.csv")


def _run_builder(monkeypatch, spec_path, output_path) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_manifest.py",
            "--spec",
            str(spec_path),
            "--output",
            str(output_path),
        ],
    )
    build_manifest.main()


def _touch_image(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake image bytes")
    return path


def _read_rows(path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as manifest_file:
        return list(csv.DictReader(manifest_file))


def _row_for_path(rows: list[dict[str, str]], path) -> dict[str, str]:
    relative_path = path.relative_to(path.parents[2]).as_posix()
    return next(row for row in rows if _normalized_path(row) == relative_path)


def _normalized_path(row: dict[str, str]) -> str:
    return row["path"].replace("\\", "/")