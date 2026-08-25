#!/usr/bin/env python3
"""Audit XITE frontalized videos against their original FVF videos."""

import argparse
import csv
import math
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
from tqdm import tqdm


CSV_COLUMNS = (
    "subject_id",
    "subject_name",
    "class_id",
    "class_name",
    "sample_id",
    "sample_name",
)
REPAIRABLE_ERROR_KINDS = {
    "missing_frontalized",
    "unreadable_frontalized",
    "invalid_frontalized_metadata",
    "too_short",
    "not_clip_aligned",
    "fps_mismatch",
}
WARNING_ISSUE_KINDS = {"frame_loss"}
XITE_ROOT = Path(__file__).resolve().parent
DEFAULT_METADATA_PATH = XITE_ROOT / "starting_point" / "samples.csv"
DEFAULT_WARNING_CSV_PATH = XITE_ROOT / "starting_point" / "warning_samples.csv"
DEFAULT_ERROR_CSV_PATH = XITE_ROOT / "starting_point" / "error_samples.csv"


@dataclass(frozen=True)
class Issue:
    kind: str
    relative_path: str
    detail: str


@dataclass(frozen=True)
class AuditResult:
    original_count: int
    frontalized_count: int
    errors: list
    warnings: list


@dataclass(frozen=True)
class VideoMetadata:
    readable: bool
    frame_count: int
    fps: float
    width: int
    height: int


def _load_repair_rows(metadata_path: Path, repair_paths: set) -> list[dict]:
    with Path(metadata_path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if tuple(reader.fieldnames or ()) != CSV_COLUMNS:
            raise ValueError(
                f"Metadata file {metadata_path} must contain columns in this order: "
                + ", ".join(CSV_COLUMNS)
            )
        rows = []
        seen_paths = set()
        matched_paths = set()
        for row in reader:
            relative_path = (
                f"{row['subject_name'].strip()}/{row['sample_name']}.mp4"
            )
            if relative_path in seen_paths:
                raise ValueError(f"Duplicate metadata sample path: {relative_path}")
            seen_paths.add(relative_path)
            if relative_path in repair_paths:
                rows.append(row)
                matched_paths.add(relative_path)
        missing_paths = sorted(repair_paths - matched_paths)
        if missing_paths:
            raise ValueError(
                "Repairable audit samples missing from metadata: "
                + ", ".join(missing_paths)
            )
    return rows


def _write_csv(output_path: Path, rows: list[dict]) -> int:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        dir=output_path.parent,
        text=True,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, output_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return len(rows)


def write_repair_csv(
    metadata_path: Path, output_path: Path, result: AuditResult
) -> int:
    repair_paths = {
        issue.relative_path
        for issue in result.errors
        if issue.kind in REPAIRABLE_ERROR_KINDS
    } | {
        issue.relative_path
        for issue in result.warnings
        if issue.kind in WARNING_ISSUE_KINDS
    }
    return _write_csv(output_path, _load_repair_rows(metadata_path, repair_paths))


def write_issue_csvs(
    metadata_path: Path,
    warning_output_path: Path,
    error_output_path: Path,
    result: AuditResult,
) -> tuple[int, int]:
    warning_paths = {
        issue.relative_path
        for issue in result.warnings
        if issue.kind in WARNING_ISSUE_KINDS
    }
    error_paths = {
        issue.relative_path
        for issue in result.errors
        if issue.kind in REPAIRABLE_ERROR_KINDS
    }
    warning_rows = _load_repair_rows(metadata_path, warning_paths)
    error_rows = _load_repair_rows(metadata_path, error_paths)
    return (
        _write_csv(warning_output_path, warning_rows),
        _write_csv(error_output_path, error_rows),
    )


def _collect_videos(root: Path) -> dict:
    return {
        path.relative_to(root).as_posix(): path
        for path in sorted(root.rglob("*.mp4"))
    }


def _load_label_paths(labels_path: Path) -> set:
    with labels_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"subject_name", "sample_name"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(
                f"Labels file {labels_path} must contain subject_name and sample_name"
            )
        return {
            f"{row['subject_name'].strip()}/{row['sample_name']}.mp4"
            for row in reader
        }


def _probe_video(path: Path) -> VideoMetadata:
    capture = cv2.VideoCapture(str(path))
    opened = capture.isOpened()
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) if opened else 0
    fps = float(capture.get(cv2.CAP_PROP_FPS)) if opened else 0.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) if opened else 0
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) if opened else 0
    decoded = False
    if opened and frame_count > 0 and width > 0 and height > 0:
        decoded, _ = capture.read()
    capture.release()
    return VideoMetadata(
        readable=bool(opened and decoded),
        frame_count=frame_count,
        fps=fps,
        width=width,
        height=height,
    )


def _invalid_metadata_detail(metadata: VideoMetadata):
    if metadata.frame_count <= 0:
        return f"frame count must be positive, found {metadata.frame_count}"
    if metadata.width <= 0 or metadata.height <= 0:
        return f"dimensions must be positive, found {metadata.width}x{metadata.height}"
    if not math.isfinite(metadata.fps) or metadata.fps <= 0:
        return f"FPS must be finite and positive, found {metadata.fps}"
    return None


def audit_videos(
    original_root: Path,
    frontalized_root: Path,
    *,
    clip_length: int = 16,
    labels_path: Path = None,
    show_progress: bool = False,
) -> AuditResult:
    """Return hard errors and quality warnings for an XITE video pair tree."""
    original_root = Path(original_root)
    frontalized_root = Path(frontalized_root)
    if not original_root.is_dir():
        raise ValueError(f"Original video root is not a directory: {original_root}")
    if not frontalized_root.is_dir():
        raise ValueError(
            f"Frontalized video root is not a directory: {frontalized_root}"
        )
    if clip_length <= 0:
        raise ValueError("clip_length must be greater than zero")

    all_originals = _collect_videos(original_root)
    all_frontalized = _collect_videos(frontalized_root)
    selected_paths = (
        _load_label_paths(Path(labels_path))
        if labels_path is not None
        else set(all_originals)
    )
    originals = {
        relative_path: all_originals[relative_path]
        for relative_path in selected_paths
        if relative_path in all_originals
    }
    frontalized = {
        relative_path: all_frontalized[relative_path]
        for relative_path in selected_paths
        if relative_path in all_frontalized
    }

    errors = []
    warnings = []
    for relative_path in sorted(selected_paths - set(all_originals)):
        errors.append(
            Issue("missing_original", relative_path, "labels reference no original video")
        )
    for relative_path in sorted(set(originals) - set(frontalized)):
        errors.append(
            Issue(
                "missing_frontalized",
                relative_path,
                "original exists but frontalized output is missing",
            )
        )
    if labels_path is None:
        for relative_path in sorted(set(all_frontalized) - set(all_originals)):
            errors.append(
                Issue(
                    "unexpected_frontalized",
                    relative_path,
                    "frontalized output has no matching original",
                )
            )

    paired_paths = sorted(set(originals) & set(frontalized))
    videos_to_check = (
        tqdm(
            paired_paths,
            desc="Checking frontalized videos",
            total=len(paired_paths),
            unit="video",
        )
        if show_progress
        else paired_paths
    )
    for relative_path in videos_to_check:
        original_metadata = _probe_video(originals[relative_path])
        frontalized_metadata = _probe_video(frontalized[relative_path])
        if not original_metadata.readable:
            errors.append(
                Issue("unreadable_original", relative_path, "original cannot be decoded")
            )
            continue
        invalid_original = _invalid_metadata_detail(original_metadata)
        if invalid_original is not None:
            errors.append(
                Issue("invalid_original_metadata", relative_path, invalid_original)
            )
            continue
        if not frontalized_metadata.readable:
            errors.append(
                Issue(
                    "unreadable_frontalized",
                    relative_path,
                    "frontalized output cannot be decoded",
                )
            )
            continue
        invalid_frontalized = _invalid_metadata_detail(frontalized_metadata)
        if invalid_frontalized is not None:
            errors.append(
                Issue(
                    "invalid_frontalized_metadata",
                    relative_path,
                    invalid_frontalized,
                )
            )
            continue

        if frontalized_metadata.frame_count < clip_length:
            errors.append(
                Issue(
                    "too_short",
                    relative_path,
                    f"{frontalized_metadata.frame_count} frames; requires at least {clip_length}",
                )
            )
        elif frontalized_metadata.frame_count % clip_length != 0:
            errors.append(
                Issue(
                    "not_clip_aligned",
                    relative_path,
                    f"{frontalized_metadata.frame_count} frames is not divisible by {clip_length}",
                )
            )
        if abs(frontalized_metadata.fps - original_metadata.fps) > 0.01:
            errors.append(
                Issue(
                    "fps_mismatch",
                    relative_path,
                    f"original FPS {original_metadata.fps:g}, frontalized FPS {frontalized_metadata.fps:g}",
                )
            )
        if frontalized_metadata.frame_count < original_metadata.frame_count:
            warnings.append(
                Issue(
                    "frame_loss",
                    relative_path,
                    f"{original_metadata.frame_count} original frames, "
                    f"{frontalized_metadata.frame_count} frontalized frames",
                )
            )

    return AuditResult(
        original_count=len(originals),
        frontalized_count=len(frontalized),
        errors=errors,
        warnings=warnings,
    )


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Check XITE frontalized videos against their originals."
    )
    parser.add_argument(
        "--original-root",
        type=Path,
        default=XITE_ROOT / "video" / "video",
        help="Original video tree (default: XITE/video/video)",
    )
    parser.add_argument(
        "--frontalized-root",
        type=Path,
        default=XITE_ROOT / "video" / "video_frontalized",
        help="Frontalized video tree (default: XITE/video/video_frontalized)",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=None,
        help="Optional tab-separated metadata file limiting the checked samples",
    )
    parser.add_argument(
        "--clip-length",
        type=int,
        default=16,
        help="Required minimum and frame-count divisor (default: 16)",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    try:
        result = audit_videos(
            args.original_root,
            args.frontalized_root,
            clip_length=args.clip_length,
            labels_path=args.labels,
            show_progress=True,
        )
        warning_count, error_count = write_issue_csvs(
            args.labels if args.labels is not None else DEFAULT_METADATA_PATH,
            DEFAULT_WARNING_CSV_PATH,
            DEFAULT_ERROR_CSV_PATH,
            result,
        )
    except (OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2

    for issue in result.errors:
        print(f"ERROR [{issue.kind}] {issue.relative_path}: {issue.detail}")
    for issue in result.warnings:
        print(f"WARNING [{issue.kind}] {issue.relative_path}: {issue.detail}")
    print(
        "Summary: "
        f"originals={result.original_count} "
        f"frontalized={result.frontalized_count} "
        f"errors={len(result.errors)} warnings={len(result.warnings)}"
    )
    print(f"Warning CSV: {DEFAULT_WARNING_CSV_PATH} samples={warning_count}")
    print(f"Error CSV: {DEFAULT_ERROR_CSV_PATH} samples={error_count}")
    return 1 if result.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
