#!/usr/bin/env python3
"""Normalize PEMF videos and generate Part-A-compatible sample metadata."""

from __future__ import annotations

import argparse
import csv
import os
import re
import tempfile
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
from xml.etree import ElementTree
from zipfile import BadZipFile, ZipFile


CSV_COLUMNS = (
    "subject_id",
    "subject_name",
    "class_id",
    "class_name",
    "sample_id",
    "sample_name",
)
EXPECTED_METADATA_ROWS = 272
EXPECTED_SAMPLE_COUNT = 277
EXPECTED_SUBJECT_COUNT = 68
SUBJECT_PATTERN = re.compile(r"S\d{3}")
CLIP_PATTERN = re.compile(r"(S\d{3})([ALNP])")
NORMALIZED_SAMPLE_PATTERN = re.compile(r"(S\d{3})([ALNP])(\d*)")
CONDITION_ORDER = {condition: index for index, condition in enumerate("ALNP")}
XML_NAMESPACE = {
    "main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "rel": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}


def _natural_key(value: str) -> tuple[object, ...]:
    return tuple(
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", value)
    )


def _column_index(cell_reference: str) -> int:
    letters = "".join(character for character in cell_reference if character.isalpha())
    index = 0
    for character in letters:
        index = index * 26 + ord(character.upper()) - ord("A") + 1
    return index - 1


def _read_xlsx_rows(workbook_path: Path, sheet_name: str) -> list[list[str]]:
    try:
        archive = ZipFile(workbook_path)
    except (FileNotFoundError, BadZipFile) as error:
        raise ValueError(f"Cannot read PEMF workbook: {workbook_path}") from error

    with archive:
        shared_strings = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in root.findall("main:si", XML_NAMESPACE):
                shared_strings.append(
                    "".join(
                        element.text or ""
                        for element in item.iterfind(".//main:t", XML_NAMESPACE)
                    )
                )

        workbook = ElementTree.fromstring(archive.read("xl/workbook.xml"))
        relationships = ElementTree.fromstring(
            archive.read("xl/_rels/workbook.xml.rels")
        )
        targets = {
            relationship.attrib["Id"]: relationship.attrib["Target"]
            for relationship in relationships
        }
        worksheet_target = None
        for sheet in workbook.findall("main:sheets/main:sheet", XML_NAMESPACE):
            if sheet.attrib["name"] == sheet_name:
                relationship_id = sheet.attrib[f"{{{XML_NAMESPACE['rel']}}}id"]
                worksheet_target = targets[relationship_id]
                break
        if worksheet_target is None:
            raise ValueError(
                f"Worksheet {sheet_name!r} not found in {workbook_path}"
            )
        if not worksheet_target.startswith("xl/"):
            worksheet_target = f"xl/{worksheet_target.lstrip('/')}"

        worksheet = ElementTree.fromstring(archive.read(worksheet_target))
        rows = []
        for row_element in worksheet.findall(
            ".//main:sheetData/main:row", XML_NAMESPACE
        ):
            values_by_column = {}
            for cell in row_element.findall("main:c", XML_NAMESPACE):
                column = _column_index(cell.attrib["r"])
                cell_type = cell.attrib.get("t")
                value_element = cell.find("main:v", XML_NAMESPACE)
                value = "" if value_element is None else value_element.text or ""
                if cell_type == "s" and value:
                    value = shared_strings[int(value)]
                elif cell_type == "inlineStr":
                    value = "".join(
                        element.text or ""
                        for element in cell.iterfind(".//main:t", XML_NAMESPACE)
                    )
                values_by_column[column] = value
            if values_by_column:
                rows.append(
                    [
                        values_by_column.get(column, "")
                        for column in range(max(values_by_column) + 1)
                    ]
                )
        return rows


def _rounded_intensity(raw_intensity: str) -> int:
    mean_text = raw_intensity.split("(", maxsplit=1)[0].strip().replace(",", ".")
    try:
        mean = Decimal(mean_text)
    except InvalidOperation as error:
        raise ValueError(f"Invalid PEMF intensity value: {raw_intensity!r}") from error
    rounded = int(mean.quantize(Decimal("1"), rounding=ROUND_HALF_UP))
    if not 0 <= rounded <= 8:
        raise ValueError(
            f"Rounded PEMF intensity must be between 0 and 8: {raw_intensity!r}"
        )
    return rounded


def read_workbook_metadata(workbook_path: str | Path) -> dict[str, dict]:
    """Return clip metadata keyed by IDs such as ``S001A``."""
    workbook_path = Path(workbook_path)
    rows = _read_xlsx_rows(workbook_path, "Articulo")
    if not rows:
        raise ValueError(f"Worksheet 'Articulo' is empty in {workbook_path}")
    headers = {name.strip(): index for index, name in enumerate(rows[0])}
    missing_headers = {"Clip", "Intensity"} - headers.keys()
    if missing_headers:
        raise ValueError(
            "Missing required PEMF workbook columns: "
            + ", ".join(sorted(missing_headers))
        )

    metadata = {}
    for excel_row, values in enumerate(rows[1:], start=2):
        clip = values[headers["Clip"]].strip()
        clip_match = CLIP_PATTERN.fullmatch(clip)
        if clip_match is None:
            raise ValueError(f"Invalid clip ID at Excel row {excel_row}: {clip!r}")
        if clip in metadata:
            raise ValueError(f"Duplicate clip ID in workbook: {clip}")
        raw_intensity = values[headers["Intensity"]]
        metadata[clip] = {
            "class_id": _rounded_intensity(raw_intensity),
            "class_name": clip_match.group(2),
        }
    return metadata


def _condition_from_folder(folder_name: str) -> str:
    normalized = re.sub(r"[^a-z]", "", folder_name.lower())
    if normalized.startswith("algom"):
        return "A"
    if normalized.startswith("laser"):
        return "L"
    if normalized.startswith("neutral"):
        return "N"
    if normalized.startswith("posed"):
        return "P"
    raise ValueError(f"Unknown PEMF condition folder: {folder_name!r}")


def _discover_sources(original_root: Path) -> dict[str, list[Path]]:
    if not original_root.is_dir():
        raise ValueError(f"Missing PEMF video directory: {original_root}")
    sources_by_clip: dict[str, list[Path]] = {}
    for video_path in original_root.glob("**/*"):
        if not video_path.is_file() or video_path.suffix.lower() != ".mp4":
            continue
        relative_parts = video_path.relative_to(original_root).parts
        if len(relative_parts) < 2:
            raise ValueError(f"Video is not inside a subject directory: {video_path}")
        subject_name = relative_parts[0]
        if SUBJECT_PATTERN.fullmatch(subject_name) is None:
            raise ValueError(f"Invalid PEMF subject directory: {subject_name!r}")

        if len(relative_parts) == 2:
            normalized_match = NORMALIZED_SAMPLE_PATTERN.fullmatch(video_path.stem)
            if normalized_match is None or normalized_match.group(1) != subject_name:
                raise ValueError(f"Invalid normalized PEMF video name: {video_path.name}")
            condition = normalized_match.group(2)
        else:
            condition = _condition_from_folder(relative_parts[1])
        clip = f"{subject_name}{condition}"
        sources_by_clip.setdefault(clip, []).append(video_path)

    if not sources_by_clip:
        raise ValueError(f"No MP4 videos found in {original_root}")
    for sources in sources_by_clip.values():
        sources.sort(key=lambda path: _natural_key(str(path.relative_to(original_root))))
    return sources_by_clip


def _build_rows(
    original_root: Path,
    metadata: dict[str, dict],
) -> tuple[list[dict], list[tuple[Path, Path]]]:
    sources_by_clip = _discover_sources(original_root)
    missing_metadata = sorted(set(sources_by_clip) - set(metadata), key=_natural_key)
    if missing_metadata:
        raise ValueError(
            "Videos have no matching workbook row: " + ", ".join(missing_metadata)
        )

    subjects = sorted(
        {clip[:4] for clip in sources_by_clip}, key=_natural_key
    )
    subject_ids = {
        subject_name: subject_id
        for subject_id, subject_name in enumerate(subjects, start=1)
    }
    rows = []
    moves = []
    sample_id = 1
    for clip in sorted(
        sources_by_clip,
        key=lambda value: (_natural_key(value[:4]), CONDITION_ORDER[value[4]]),
    ):
        sources = sources_by_clip[clip]
        normalized_sources = [source for source in sources if source.parent == original_root / clip[:4]]
        nested_sources = [source for source in sources if source.parent != original_root / clip[:4]]
        if normalized_sources and nested_sources:
            raise ValueError(
                "Refusing to overwrite conflicting video path: "
                f"{normalized_sources[0]}"
            )

        for duplicate_index, source_path in enumerate(sources, start=1):
            sample_name = clip if len(sources) == 1 else f"{clip}{duplicate_index}"
            destination = original_root / clip[:4] / f"{sample_name}.mp4"
            if source_path != destination:
                if destination.exists() or destination.is_symlink():
                    raise ValueError(
                        f"Refusing to overwrite conflicting video path: {destination}"
                    )
                moves.append((source_path, destination))
            rows.append(
                {
                    "subject_id": subject_ids[clip[:4]],
                    "subject_name": clip[:4],
                    "class_id": metadata[clip]["class_id"],
                    "class_name": metadata[clip]["class_name"],
                    "sample_id": sample_id,
                    "sample_name": sample_name,
                }
            )
            sample_id += 1
    return rows, moves


def _validate_expected_dataset(rows: list[dict], metadata: dict[str, dict]) -> None:
    found_subjects = {row["subject_name"] for row in rows}
    found_clips = {
        f'{row["subject_name"]}{row["class_name"]}' for row in rows
    }
    if len(metadata) != EXPECTED_METADATA_ROWS:
        raise ValueError(
            "Unexpected workbook row count: "
            f"expected {EXPECTED_METADATA_ROWS}, found {len(metadata)}"
        )
    if len(rows) != EXPECTED_SAMPLE_COUNT:
        raise ValueError(
            "Unexpected video count: "
            f"expected {EXPECTED_SAMPLE_COUNT}, found {len(rows)}"
        )
    if len(found_subjects) != EXPECTED_SUBJECT_COUNT:
        raise ValueError(
            "Unexpected subject count: "
            f"expected {EXPECTED_SUBJECT_COUNT}, found {len(found_subjects)}"
        )
    missing_clips = sorted(set(metadata) - found_clips, key=_natural_key)
    if missing_clips:
        raise ValueError(
            "Workbook clips have no matching video: " + ", ".join(missing_clips)
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
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _move_videos(moves: list[tuple[Path, Path]]) -> list[tuple[Path, Path]]:
    completed = []
    try:
        for source, destination in moves:
            destination.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(source, destination)
            except FileExistsError as error:
                raise ValueError(
                    f"Refusing to overwrite conflicting video path: {destination}"
                ) from error
            try:
                source.unlink()
            except BaseException:
                destination.unlink(missing_ok=True)
                raise
            completed.append((source, destination))
    except BaseException:
        _rollback_moves(completed)
        raise
    return completed


def _rollback_moves(completed: list[tuple[Path, Path]]) -> None:
    for source, destination in reversed(completed):
        source.parent.mkdir(parents=True, exist_ok=True)
        os.link(destination, source)
        destination.unlink()


def _remove_empty_source_directories(
    original_root: Path, moves: list[tuple[Path, Path]]
) -> None:
    candidates = {
        directory
        for source, _ in moves
        for directory in source.parents
        if directory != original_root and original_root in directory.parents
    }
    for directory in sorted(candidates, key=lambda path: len(path.parts), reverse=True):
        if SUBJECT_PATTERN.fullmatch(directory.name):
            continue
        try:
            directory.rmdir()
        except OSError:
            pass


def prepare_dataset(
    pemf_root: str | Path,
    *,
    metadata_path: str | Path | None = None,
    validate_expected_counts: bool = True,
    dry_run: bool = False,
) -> dict[str, object]:
    """Normalize PEMF videos in place and write ``starting_point/samples.csv``."""
    pemf_root = Path(pemf_root).resolve()
    workbook_path = (
        Path(metadata_path).resolve()
        if metadata_path is not None
        else pemf_root / "PEMF_Database.xlsx"
    )
    original_root = pemf_root / "video" / "Original"
    csv_path = pemf_root / "starting_point" / "samples.csv"
    metadata = read_workbook_metadata(workbook_path)
    rows, moves = _build_rows(original_root, metadata)
    if validate_expected_counts:
        _validate_expected_dataset(rows, metadata)

    if not dry_run:
        completed = _move_videos(moves)
        try:
            _write_tsv(csv_path, rows)
        except BaseException:
            _rollback_moves(completed)
            raise
        _remove_empty_source_directories(original_root, moves)

    return {
        "csv_path": csv_path,
        "sample_count": len(rows),
        "moved_count": len(moves),
        "dry_run": dry_run,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Normalize PEMF videos in place and generate a Part-A-compatible "
            "starting_point/samples.csv file."
        )
    )
    parser.add_argument(
        "--pemf-root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="PEMF dataset root (default: directory containing this script)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and display the planned counts without changing any files",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    result = prepare_dataset(args.pemf_root, dry_run=args.dry_run)
    action = "Would move" if args.dry_run else "Moved"
    print(
        f"Validated {result['sample_count']} samples. "
        f"{action} {result['moved_count']} videos."
    )
    if args.dry_run:
        print("Dry run only: no videos or CSV files were changed.")
    else:
        print(f"Wrote {result['csv_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
