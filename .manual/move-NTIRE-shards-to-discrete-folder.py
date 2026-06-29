from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path


DEFAULT_SHARD_DIR = Path(r"D:\aigc-dataset\shard_0\shard_0")
IMAGE_SUFFIXES = {".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Move shard images into fake/real folders using labels.csv."
    )
    parser.add_argument(
        "--shard-dir",
        type=Path,
        default=DEFAULT_SHARD_DIR,
        help="Shard directory containing labels.csv and usually an images/ folder.",
    )
    parser.add_argument(
        "--labels-csv",
        type=Path,
        default=None,
        help="Path to labels.csv. Defaults to <shard-dir>/labels.csv.",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=None,
        help="Directory containing images. Defaults to <shard-dir>/images when present, else <shard-dir>.",
    )
    parser.add_argument("--real-dir-name", default="real")
    parser.add_argument("--fake-dir-name", default="fake")
    parser.add_argument("--real-label", default="0")
    parser.add_argument("--fake-label", default="1")
    parser.add_argument(
        "--on-existing",
        choices=["error", "skip", "overwrite"],
        default="error",
        help="What to do when the destination file already exists.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print moves without changing files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    shard_dir = args.shard_dir
    labels_csv = args.labels_csv or shard_dir / "labels.csv"
    images_dir = args.images_dir or _default_images_dir(shard_dir)

    if not labels_csv.is_file():
        raise SystemExit(f"Labels CSV not found: {labels_csv}")
    if not images_dir.is_dir():
        raise SystemExit(f"Images directory not found: {images_dir}")

    destinations = {
        str(args.real_label): shard_dir / args.real_dir_name,
        str(args.fake_label): shard_dir / args.fake_dir_name,
    }
    for destination_dir in destinations.values():
        if not args.dry_run:
            destination_dir.mkdir(parents=True, exist_ok=True)

    moved = 0
    skipped = 0
    missing = 0

    for row_number, image_name, label in _iter_labels(labels_csv):
        destination_dir = destinations.get(label)
        if destination_dir is None:
            skipped += 1
            print(f"Skipping row {row_number}: unsupported label {label!r} for {image_name}")
            continue

        source_path = images_dir / image_name
        if not source_path.is_file():
            missing += 1
            print(f"Missing image for row {row_number}: {source_path}")
            continue
        if source_path.suffix.lower() not in IMAGE_SUFFIXES:
            skipped += 1
            print(f"Skipping row {row_number}: not a known image suffix: {source_path}")
            continue

        destination_path = destination_dir / source_path.name
        result = _move_image(source_path, destination_path, args.on_existing, args.dry_run)
        if result == "moved":
            moved += 1
        elif result == "skipped":
            skipped += 1

    action = "Would move" if args.dry_run else "Moved"
    print(f"{action} {moved} images. Skipped {skipped}. Missing {missing}.")


def _default_images_dir(shard_dir: Path) -> Path:
    images_dir = shard_dir / "images"
    if images_dir.is_dir():
        return images_dir
    return shard_dir


def _iter_labels(labels_csv: Path):
    with labels_csv.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise SystemExit(f"Labels CSV is empty: {labels_csv}")
        if "image_name" not in reader.fieldnames or "label" not in reader.fieldnames:
            raise SystemExit("Labels CSV must contain image_name and label columns.")

        for row_number, row in enumerate(reader, start=2):
            image_name = (row.get("image_name") or "").strip()
            label = (row.get("label") or "").strip()
            if not image_name:
                print(f"Skipping row {row_number}: empty image_name")
                continue
            yield row_number, image_name, label


def _move_image(
    source_path: Path,
    destination_path: Path,
    on_existing: str,
    dry_run: bool,
) -> str:
    if destination_path.exists():
        if on_existing == "skip":
            print(f"Skipping existing destination: {destination_path}")
            return "skipped"
        if on_existing == "error":
            raise SystemExit(f"Destination already exists: {destination_path}")
        if not dry_run:
            destination_path.unlink()

    print(f"{source_path} -> {destination_path}")
    if not dry_run:
        shutil.move(str(source_path), str(destination_path))
    return "moved"


if __name__ == "__main__":
    main()