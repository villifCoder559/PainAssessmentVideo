#!/usr/bin/env python3
"""Prepare XITE challenge metadata for the repository's video pipelines."""

from __future__ import annotations

import argparse
import csv
import os
import re
import tempfile
from pathlib import Path


CSV_COLUMNS = (
    "subject_id",
    "subject_name",
    "class_id",
    "class_name",
    "sample_id",
    "sample_name",
)
EXPECTED_CLASS_COUNTS = {0: ("PL1", 1557), 1: ("PL2", 1560)}
EXPECTED_TRAIN_SUBJECTS = 26
EXPECTED_TEST_SUBJECTS = 4
EXPECTED_TEST_SAMPLES = 480


def _natural_key(value: str) -> tuple[object, ...]:
    return tuple(
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", value)
    )


def _video_names(video_path: Path) -> tuple[str, str]:
    subject_name = video_path.parent.name.strip()
    sample_name = video_path.stem
    if not subject_name or re.fullmatch(
        rf"{re.escape(subject_name)}_\d+", sample_name
    ) is None:
        raise ValueError(
            "Expected '<subject>_<numeric-segment>.mp4' but found "
            f"{video_path}"
        )
    return subject_name, sample_name


def _validate_unique_keys(rows: list[dict], split_name: str) -> None:
    seen = set()
    for row in rows:
        key = (row["subject_name"], row["sample_name"])
        if key in seen:
            raise ValueError(
                f"Duplicate {split_name} sample key: {key[0]}/{key[1]}"
            )
        seen.add(key)


def _discover_video_rows(xite_root: Path) -> tuple[list[dict], list[dict]]:
    train_sources = (
        (0, "PL1", xite_root / "train_data" / "low_pain" / "fvf"),
        (1, "PL2", xite_root / "train_data" / "med_pain" / "fvf"),
    )
    train_rows = []
    for class_id, class_name, source_root in train_sources:
        if not source_root.is_dir():
            raise ValueError(f"Missing required FVF directory: {source_root}")
        for video_path in source_root.glob("*/*.mp4"):
            subject_name, sample_name = _video_names(video_path)
            train_rows.append(
                {
                    "subject_name": subject_name,
                    "class_id": class_id,
                    "class_name": class_name,
                    "sample_name": sample_name,
                    "source_path": video_path,
                }
            )

    test_root = xite_root / "test_data" / "fvf"
    if not test_root.is_dir():
        raise ValueError(f"Missing required FVF directory: {test_root}")
    test_rows = []
    for video_path in test_root.glob("*/*.mp4"):
        subject_name, sample_name = _video_names(video_path)
        test_rows.append(
            {
                "subject_name": subject_name,
                "class_id": -1,
                "class_name": "UNKNOWN",
                "sample_name": sample_name,
                "source_path": video_path,
            }
        )

    _validate_unique_keys(train_rows, "train")
    _validate_unique_keys(test_rows, "test")
    train_rows.sort(
        key=lambda row: (
            _natural_key(row["subject_name"]),
            row["class_id"],
            _natural_key(row["sample_name"]),
        )
    )
    test_rows.sort(
        key=lambda row: (
            _natural_key(row["subject_name"]),
            _natural_key(row["sample_name"]),
        )
    )
    return train_rows, test_rows


def _assign_ids(train_rows: list[dict], test_rows: list[dict]) -> None:
    train_subjects = sorted(
        {row["subject_name"] for row in train_rows}, key=_natural_key
    )
    test_subjects = sorted(
        {row["subject_name"] for row in test_rows}, key=_natural_key
    )
    overlapping_subjects = sorted(
        set(train_subjects) & set(test_subjects), key=_natural_key
    )
    if overlapping_subjects:
        raise ValueError(
            "Subjects occur in both train and test: "
            + ", ".join(overlapping_subjects)
        )
    subject_ids = {
        subject_name: subject_id
        for subject_id, subject_name in enumerate(
            train_subjects + test_subjects, start=1
        )
    }

    for sample_id, row in enumerate(train_rows + test_rows, start=1):
        row["subject_id"] = subject_ids[row["subject_name"]]
        row["sample_id"] = sample_id


def _validate_expected_counts(train_rows: list[dict], test_rows: list[dict]) -> None:
    for class_id, (class_name, expected_count) in EXPECTED_CLASS_COUNTS.items():
        found_count = sum(row["class_id"] == class_id for row in train_rows)
        if found_count != expected_count:
            raise ValueError(
                f"Unexpected {class_name} sample count: expected {expected_count}, "
                f"found {found_count}"
            )

    train_subjects = {row["subject_name"] for row in train_rows}
    test_subjects = {row["subject_name"] for row in test_rows}
    if len(train_subjects) != EXPECTED_TRAIN_SUBJECTS:
        raise ValueError(
            "Unexpected train subject count: "
            f"expected {EXPECTED_TRAIN_SUBJECTS}, found {len(train_subjects)}"
        )
    if len(test_subjects) != EXPECTED_TEST_SUBJECTS:
        raise ValueError(
            "Unexpected test subject count: "
            f"expected {EXPECTED_TEST_SUBJECTS}, found {len(test_subjects)}"
        )
    if len(test_rows) != EXPECTED_TEST_SAMPLES:
        raise ValueError(
            "Unexpected test sample count: "
            f"expected {EXPECTED_TEST_SAMPLES}, found {len(test_rows)}"
        )


def _write_tsv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, delimiter="\t")
            writer.writeheader()
            for row in rows:
                writer.writerow({column: row[column] for column in CSV_COLUMNS})
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _organize_videos(xite_root: Path, rows: list[dict]) -> int:
    video_root = xite_root / "video" / "video"
    links_to_create = []
    for row in rows:
        destination = (
            video_root
            / row["subject_name"]
            / f'{row["sample_name"]}.mp4'
        )
        if (
            destination.is_symlink()
            and destination.resolve() == row["source_path"].resolve()
        ):
            continue
        if destination.exists() or destination.is_symlink():
            raise ValueError(
                f"Refusing to overwrite conflicting video path: {destination}"
            )
        links_to_create.append((destination, row["source_path"]))

    for destination, source_path in links_to_create:
        destination.parent.mkdir(parents=True, exist_ok=True)
        relative_source = os.path.relpath(source_path, destination.parent)
        destination.symlink_to(relative_source)
    return len(rows)


def prepare_dataset(
    xite_root: str | Path,
    *,
    validate_expected_counts: bool = True,
    reorganize_videos: bool = False,
) -> dict[str, Path]:
    """Generate Part-A-compatible train, test, and combined XITE TSV files."""
    xite_root = Path(xite_root).resolve()
    train_rows, test_rows = _discover_video_rows(xite_root)
    _assign_ids(train_rows, test_rows)
    if validate_expected_counts:
        _validate_expected_counts(train_rows, test_rows)

    output_dir = xite_root / "starting_point"
    outputs = {
        "train": output_dir / "train_samples.csv",
        "test": output_dir / "test_samples.csv",
        "all": output_dir / "samples.csv",
    }
    _write_tsv(outputs["train"], train_rows)
    _write_tsv(outputs["test"], test_rows)
    _write_tsv(outputs["all"], train_rows + test_rows)
    if reorganize_videos:
        _organize_videos(xite_root, train_rows + test_rows)
    return outputs


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Part-A-compatible metadata for the XITE challenge."
    )
    parser.add_argument(
        "--xite-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="XITE dataset root (default: directory containing this starting_point folder)",
    )
    parser.add_argument(
        "--reorganize-videos",
        action="store_true",
        help=(
            "Create relative FVF symlinks under "
            "<xite-root>/video/video/<subject>/<sample>.mp4"
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    outputs = prepare_dataset(
        args.xite_root,
        reorganize_videos=args.reorganize_videos,
    )
    print("Generated 3117 train, 480 test, and 3597 combined rows.")
    for split_name in ("train", "test", "all"):
        print(f"  {split_name}: {outputs[split_name]}")
    if args.reorganize_videos:
        video_root = Path(args.xite_root).resolve() / "video" / "video"
        print(f"Prepared 3597 relative video symlinks in {video_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
