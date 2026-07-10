from __future__ import annotations

import argparse
import csv
import hashlib
import math
import random
import sys
from collections import defaultdict
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from micv.data.labels import FAKE_CLASS_NAMES, REAL_CLASS_NAMES  # noqa: E402

IMAGE_SUFFIXES = {".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class ManifestCandidate:
    path: Path
    label: int
    split: str
    generator: str
    source_tier: str = ""
    source_dataset: str = ""
    task_type: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a MICV manifest from image folders.")
    parser.add_argument("--root", default=None, help="Dataset root containing split/class folders.")
    parser.add_argument("--output", required=True, help="CSV manifest path to write.")
    parser.add_argument(
        "--spec",
        default=None,
        help="YAML hybrid manifest spec with explicit split sources and random-split sources.",
    )
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument(
        "--layout",
        choices=["auto", "split-class", "class-split", "binary-dirs"],
        default="auto",
        help=(
            "Dataset layout. split-class means root/train/real. class-split means "
            "root/real/train. binary-dirs uses root/real and root/fake or explicit dirs."
        ),
    )
    parser.add_argument("--real-dirs", nargs="*", default=None, help="Explicit real image directories.")
    parser.add_argument("--fake-dirs", nargs="*", default=None, help="Explicit AI/generated image directories.")
    parser.add_argument("--binary-split", default="train", help="Split name for binary-dirs layout.")
    parser.add_argument("--real-names", nargs="+", default=list(REAL_CLASS_NAMES))
    parser.add_argument("--fake-names", nargs="+", default=list(FAKE_CLASS_NAMES))
    parser.add_argument("--hash", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--dimensions", action="store_true", help="Record image width and height.")
    parser.add_argument(
        "--no-verify-images",
        action="store_false",
        dest="verify_images",
        help="Do not load images before writing the manifest.",
    )
    parser.add_argument("--absolute-paths", action="store_true", help="Write absolute paths instead of root-relative paths.")
    parser.add_argument("--seed", type=int, default=42, help="Seed used by hybrid random splitting and sampling.")
    parser.add_argument(
        "--random-train-frac",
        type=float,
        default=0.8,
        help="Fraction of hybrid random candidates assigned to train.",
    )
    parser.add_argument(
        "--random-split-unit",
        choices=["image", "leaf-folder"],
        default="image",
        help="Unit used when assigning hybrid random candidates to train/val.",
    )
    parser.add_argument("--source-tier", default="custom")
    parser.add_argument("--source-dataset", default="custom")
    parser.set_defaults(verify_images=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec = _load_spec(args.spec) if args.spec else None
    root = _manifest_root(args, spec)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    candidates = list(_iter_spec_candidates(args, root, spec) if spec is not None else _iter_candidates(args, root))
    if args.verify_images:
        candidates = _filter_verified_candidates(candidates)
    candidates = _deduplicate_candidates(candidates)
    if not candidates:
        raise SystemExit(
            "No images found. Check --root, --layout, --splits, --real-names/--fake-names, "
            "or provide --real-dirs and --fake-dirs."
        )

    fieldnames = [
        "path",
        "label",
        "split",
        "source_tier",
        "source_dataset",
        "generator",
        "task_type",
        "width",
        "height",
        "md5",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for candidate in candidates:
            width, height = _dimensions(candidate.path) if args.dimensions else ("", "")
            writer.writerow(
                {
                    "path": _format_path(candidate.path, root, absolute=args.absolute_paths),
                    "label": candidate.label,
                    "split": candidate.split,
                    "source_tier": candidate.source_tier or args.source_tier,
                    "source_dataset": candidate.source_dataset or args.source_dataset,
                    "generator": candidate.generator,
                    "task_type": candidate.task_type,
                    "width": width,
                    "height": height,
                    "md5": _md5(candidate.path),
                }
            )

    print(f"Wrote {len(candidates)} rows to {output_path}")


def _iter_candidates(args: argparse.Namespace, root: Path) -> Iterator[ManifestCandidate]:
    real_names = _normalize_names(args.real_names)
    fake_names = _normalize_names(args.fake_names)
    layout = _resolve_layout(args, root, real_names, fake_names)

    if layout == "split-class":
        yield from _iter_split_class(root, args.splits, real_names, fake_names)
    elif layout == "class-split":
        yield from _iter_class_split(root, args.splits, real_names, fake_names)
    elif layout == "binary-dirs":
        yield from _iter_binary_dirs(args, root)
    else:
        raise ValueError(f"Unsupported layout: {layout}")


def _load_spec(spec_path: str | None) -> dict[str, Any] | None:
    if spec_path is None:
        return None
    with Path(spec_path).open("r", encoding="utf-8") as spec_file:
        spec = yaml.safe_load(spec_file) or {}
    if not isinstance(spec, dict):
        raise ValueError("Hybrid manifest spec must be a YAML mapping.")
    return spec


def _manifest_root(args: argparse.Namespace, spec: dict[str, Any] | None) -> Path:
    if spec is not None:
        return Path(spec.get("root") or args.root or ".")
    if args.root is None:
        raise SystemExit("--root is required unless --spec is provided.")
    return Path(args.root)


def _iter_spec_candidates(
    args: argparse.Namespace,
    root: Path,
    spec: dict[str, Any],
) -> Iterator[ManifestCandidate]:
    sources = spec.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("Hybrid manifest spec must include a non-empty 'sources' list.")

    seed = int(spec.get("seed", args.seed))
    train_fraction = _bounded_fraction(spec.get("random_train_fraction", args.random_train_frac))
    split_unit = str(spec.get("random_split_unit", args.random_split_unit))
    if split_unit not in {"image", "leaf-folder"}:
        raise ValueError("random_split_unit must be either 'image' or 'leaf-folder'.")

    rng = random.Random(seed)
    random_candidates: list[ManifestCandidate] = []

    for source in sources:
        if not isinstance(source, dict):
            raise ValueError("Each hybrid manifest source must be a YAML mapping.")
        source_candidates = _candidates_from_spec_source(source, root, rng)
        split = str(source.get("split", "random")).lower()
        if split == "random":
            random_candidates.extend(source_candidates)
            continue
        for candidate in source_candidates:
            yield ManifestCandidate(
                path=candidate.path,
                label=candidate.label,
                split=split,
                generator=candidate.generator,
                source_tier=candidate.source_tier,
                source_dataset=candidate.source_dataset,
                task_type=candidate.task_type,
            )

    yield from _assign_random_split(random_candidates, train_fraction, split_unit, rng)


def _candidates_from_spec_source(
    source: dict[str, Any],
    root: Path,
    rng: random.Random,
) -> list[ManifestCandidate]:
    if "path" not in source:
        raise ValueError("Each hybrid manifest source must include a path.")
    if "label" not in source:
        raise ValueError(f"Hybrid manifest source {source['path']!r} must include a label.")

    source_dir = _resolve_input_dir(str(source["path"]), root)
    label = _normalize_spec_label(source["label"])
    split = str(source.get("split", "random")).lower()
    keep_percent = float(source.get("keep_percent", 100.0))
    min_per_leaf_folder = int(source.get("min_per_leaf_folder", 1))
    generator = str(source.get("generator", source_dir.name if label == 1 else ""))
    source_tier = str(source.get("source_tier", ""))
    source_dataset = str(source.get("source_dataset", ""))
    task_type = str(source.get("task_type", ""))

    if split not in {"random", "train", "val", "test"}:
        raise ValueError(f"Unsupported split {split!r} for hybrid manifest source {source_dir}.")
    if not source_dir.exists():
        print(f"Warning: skipping missing source: {source_dir}")
        return []

    image_paths = _sample_images_by_leaf(source_dir, keep_percent, min_per_leaf_folder, rng)
    return [
        ManifestCandidate(
            path=image_path,
            label=label,
            split=split,
            generator=generator,
            source_tier=source_tier,
            source_dataset=source_dataset,
            task_type=task_type,
        )
        for image_path in image_paths
    ]


def _sample_images_by_leaf(
    directory: Path,
    keep_percent: float,
    min_per_leaf_folder: int,
    rng: random.Random,
) -> list[Path]:
    keep_fraction = _bounded_fraction(keep_percent / 100.0)
    if keep_fraction <= 0.0:
        return []

    grouped_paths: dict[Path, list[Path]] = defaultdict(list)
    for image_path in _iter_image_paths(directory):
        grouped_paths[image_path.parent].append(image_path)

    sampled_paths: list[Path] = []
    for folder in sorted(grouped_paths):
        folder_paths = sorted(grouped_paths[folder])
        sample_count = math.ceil(len(folder_paths) * keep_fraction)
        sample_count = max(min_per_leaf_folder, sample_count)
        sample_count = min(len(folder_paths), sample_count)
        rng.shuffle(folder_paths)
        sampled_paths.extend(folder_paths[:sample_count])
    return sampled_paths


def _iter_image_paths(directory: Path) -> Iterator[Path]:
    for image_path in sorted(directory.rglob("*")):
        if image_path.is_file() and image_path.suffix.lower() in IMAGE_SUFFIXES:
            yield image_path


def _assign_random_split(
    candidates: list[ManifestCandidate],
    train_fraction: float,
    split_unit: str,
    rng: random.Random,
) -> Iterator[ManifestCandidate]:
    if not candidates:
        return

    if split_unit == "image":
        shuffled_candidates = list(candidates)
        rng.shuffle(shuffled_candidates)
        train_count = round(len(shuffled_candidates) * train_fraction)
        for index, candidate in enumerate(shuffled_candidates):
            yield _candidate_with_split(candidate, "train" if index < train_count else "val")
        return

    groups: dict[Path, list[ManifestCandidate]] = defaultdict(list)
    for candidate in candidates:
        groups[candidate.path.parent].append(candidate)

    group_items = sorted(groups.items(), key=lambda item: str(item[0]))
    rng.shuffle(group_items)
    target_train_count = round(len(candidates) * train_fraction)
    current_train_count = 0
    for _, group_candidates in group_items:
        split = "train" if current_train_count < target_train_count else "val"
        if split == "train":
            current_train_count += len(group_candidates)
        for candidate in group_candidates:
            yield _candidate_with_split(candidate, split)


def _candidate_with_split(candidate: ManifestCandidate, split: str) -> ManifestCandidate:
    return ManifestCandidate(
        path=candidate.path,
        label=candidate.label,
        split=split,
        generator=candidate.generator,
        source_tier=candidate.source_tier,
        source_dataset=candidate.source_dataset,
        task_type=candidate.task_type,
    )


def _normalize_spec_label(value: Any) -> int:
    normalized = str(value).strip().lower()
    if normalized in {"0", "real", "human", "natural", "authentic", "negative"}:
        return 0
    if normalized in {"1", "fake", "ai", "aigc", "generated", "synthetic", "positive"}:
        return 1
    raise ValueError(f"Unsupported hybrid manifest label: {value!r}")


def _bounded_fraction(value: Any) -> float:
    fraction = float(value)
    return max(0.0, min(1.0, fraction))


def _deduplicate_candidates(candidates: list[ManifestCandidate]) -> list[ManifestCandidate]:
    seen: dict[Path, ManifestCandidate] = {}
    for candidate in candidates:
        normalized_path = candidate.path.resolve()
        if normalized_path in seen:
            previous = seen[normalized_path]
            raise ValueError(
                "Duplicate image path in manifest sources: "
                f"{candidate.path} appears in split={previous.split!r} and split={candidate.split!r}."
            )
        seen[normalized_path] = candidate
    return candidates


def _filter_verified_candidates(candidates: list[ManifestCandidate]) -> list[ManifestCandidate]:
    verified_candidates: list[ManifestCandidate] = []
    skipped: list[tuple[Path, Exception]] = []
    for candidate in candidates:
        try:
            _verify_image(candidate.path)
        except Exception as error:
            skipped.append((candidate.path, error))
            continue
        verified_candidates.append(candidate)
    if skipped:
        print(f"Warning: skipped {len(skipped)} unloadable image(s).")
        for image_path, error in skipped[:5]:
            print(f"Warning: skipped {image_path}: {error}")
        if len(skipped) > 5:
            print(f"Warning: skipped {len(skipped) - 5} additional unloadable image(s).")
    return verified_candidates


def _resolve_layout(
    args: argparse.Namespace,
    root: Path,
    real_names: set[str],
    fake_names: set[str],
) -> str:
    if args.layout != "auto":
        return args.layout
    if args.real_dirs or args.fake_dirs:
        return "binary-dirs"
    if any((root / split).is_dir() for split in args.splits):
        return "split-class"
    if any((root / class_name).is_dir() for class_name in real_names | fake_names):
        class_dirs = [root / class_name for class_name in real_names | fake_names]
        if any((class_dir / split).is_dir() for class_dir in class_dirs for split in args.splits):
            return "class-split"
        return "binary-dirs"
    return "split-class"


def _iter_split_class(
    root: Path,
    splits: Sequence[str],
    real_names: set[str],
    fake_names: set[str],
) -> Iterator[ManifestCandidate]:
    for split in splits:
        split_dir = root / split
        if not split_dir.exists():
            continue
        for class_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
            label = _label_for_class(class_dir.name, real_names, fake_names)
            if label is None:
                continue
            generator = class_dir.name if label == 1 else ""
            yield from _iter_images(class_dir, label=label, split=split, generator=generator)


def _iter_class_split(
    root: Path,
    splits: Sequence[str],
    real_names: set[str],
    fake_names: set[str],
) -> Iterator[ManifestCandidate]:
    for class_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        label = _label_for_class(class_dir.name, real_names, fake_names)
        if label is None:
            continue
        for split in splits:
            split_dir = class_dir / split
            if not split_dir.exists():
                continue
            generator = class_dir.name if label == 1 else ""
            yield from _iter_images(split_dir, label=label, split=split, generator=generator)


def _iter_binary_dirs(args: argparse.Namespace, root: Path) -> Iterator[ManifestCandidate]:
    real_dirs = _explicit_or_default_dirs(args.real_dirs, root, args.real_names)
    fake_dirs = _explicit_or_default_dirs(args.fake_dirs, root, args.fake_names)
    for real_dir in real_dirs:
        yield from _iter_images(real_dir, label=0, split=args.binary_split, generator="")
    for fake_dir in fake_dirs:
        yield from _iter_images(fake_dir, label=1, split=args.binary_split, generator=fake_dir.name)


def _iter_images(directory: Path, label: int, split: str, generator: str) -> Iterator[ManifestCandidate]:
    for image_path in _iter_image_paths(directory):
        yield ManifestCandidate(path=image_path, label=label, split=split, generator=generator)


def _explicit_or_default_dirs(explicit_dirs: Sequence[str] | None, root: Path, names: Sequence[str]) -> list[Path]:
    if explicit_dirs:
        return [_resolve_input_dir(directory, root) for directory in explicit_dirs]
    return [root / name for name in names if (root / name).is_dir()]


def _resolve_input_dir(directory: str, root: Path) -> Path:
    input_dir = Path(directory)
    if input_dir.is_absolute():
        return input_dir
    root_relative = root / input_dir
    if root_relative.exists():
        return root_relative
    return input_dir


def _format_path(image_path: Path, root: Path, absolute: bool) -> str:
    if absolute:
        return str(image_path)
    try:
        return str(image_path.relative_to(root))
    except ValueError:
        return str(image_path)


def _normalize_names(names: Iterable[str]) -> set[str]:
    return {name.lower() for name in names}


def _label_for_class(class_name: str, real_names: set[str], fake_names: set[str]) -> int | None:
    normalized = class_name.lower()
    if normalized in real_names:
        return 0
    if normalized in fake_names:
        return 1
    return None


def _dimensions(image_path: Path) -> tuple[int, int]:
    with Image.open(image_path) as image:
        return image.size


def _verify_image(image_path: Path) -> None:
    with Image.open(image_path) as image:
        image.load()


def _md5(image_path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with image_path.open("rb") as image_file:
        for chunk in iter(lambda: image_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()