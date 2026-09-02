import csv
import importlib.util
from pathlib import Path

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPOSITORY_ROOT / "PEMF" / "prepare_pemf.py"
WORKBOOK_PATH = REPOSITORY_ROOT / "PEMF" / "PEMF_Database.xlsx"
EXPECTED_COLUMNS = [
    "subject_id",
    "subject_name",
    "class_id",
    "class_name",
    "sample_id",
    "sample_name",
]


def _load_prepare_pemf():
    assert SCRIPT_PATH.is_file(), f"Missing preparation script: {SCRIPT_PATH}"
    spec = importlib.util.spec_from_file_location("prepare_pemf", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _video(root: Path, relative_path: str, contents: bytes = b"video") -> Path:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(contents)
    return path


def _read_tsv(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return reader.fieldnames, list(reader)


def test_workbook_metadata_uses_clip_suffix_and_half_up_intensity_rounding():
    prepare_pemf = _load_prepare_pemf()

    metadata = prepare_pemf.read_workbook_metadata(WORKBOOK_PATH)

    assert len(metadata) == 272
    assert metadata["S001A"] == {"class_id": 5, "class_name": "A"}
    assert metadata["S029L"] == {"class_id": 6, "class_name": "L"}
    # The workbook Kind value is "Laser", but the clip suffix correctly says Posed.
    assert metadata["S051P"] == {"class_id": 2, "class_name": "P"}


def test_workbook_metadata_can_keep_intensity_as_two_decimal_float():
    prepare_pemf = _load_prepare_pemf()

    metadata = prepare_pemf.read_workbook_metadata(
        WORKBOOK_PATH, class_id_type="float"
    )

    assert str(metadata["S001A"]["class_id"]) == "5.20"
    assert str(metadata["S029L"]["class_id"]) == "5.62"


def test_prepare_dataset_moves_and_renames_duplicates_and_writes_compatible_csv(
    tmp_path,
):
    prepare_pemf = _load_prepare_pemf()
    pemf_root = tmp_path / "PEMF"
    original = pemf_root / "video" / "Original"
    _video(original, "S001/Algometer Pain/Algometer Pain S001.mp4", b"one")
    _video(original, "S019/Algometer Pain/Algometer Pain-2 S019.mp4", b"two")
    _video(original, "S019/Algometer Pain/Algometer Pain-1 S019.mp4", b"three")

    result = prepare_pemf.prepare_dataset(
        pemf_root,
        metadata_path=WORKBOOK_PATH,
        validate_expected_counts=False,
    )

    assert result["sample_count"] == 3
    assert result["moved_count"] == 3
    assert (original / "S001" / "S001A.mp4").read_bytes() == b"one"
    assert (original / "S019" / "S019A1.mp4").read_bytes() == b"three"
    assert (original / "S019" / "S019A2.mp4").read_bytes() == b"two"
    assert not (original / "S001" / "Algometer Pain").exists()
    assert not (original / "S019" / "Algometer Pain").exists()

    columns, rows = _read_tsv(pemf_root / "starting_point" / "samples.csv")
    assert columns == EXPECTED_COLUMNS
    assert rows == [
        {
            "subject_id": "1",
            "subject_name": "S001",
            "class_id": "5",
            "class_name": "A",
            "sample_id": "1",
            "sample_name": "S001A",
        },
        {
            "subject_id": "2",
            "subject_name": "S019",
            "class_id": "4",
            "class_name": "A",
            "sample_id": "2",
            "sample_name": "S019A1",
        },
        {
            "subject_id": "2",
            "subject_name": "S019",
            "class_id": "4",
            "class_name": "A",
            "sample_id": "3",
            "sample_name": "S019A2",
        },
    ]
    for row in rows:
        expected_video = (
            original / row["subject_name"] / f'{row["sample_name"]}.mp4'
        )
        assert expected_video.is_file()
        assert row["class_id"].isdigit()


def test_prepare_dataset_accepts_the_source_algomer_typo(tmp_path):
    prepare_pemf = _load_prepare_pemf()
    pemf_root = tmp_path / "PEMF"
    original = pemf_root / "video" / "Original"
    _video(original, "S013/Algomer Pain/S013A.mp4")

    prepare_pemf.prepare_dataset(
        pemf_root,
        metadata_path=WORKBOOK_PATH,
        validate_expected_counts=False,
    )

    assert (original / "S013" / "S013A.mp4").is_file()
    assert not (original / "S013" / "Algomer Pain").exists()


def test_dry_run_does_not_move_videos_or_write_csv(tmp_path):
    prepare_pemf = _load_prepare_pemf()
    pemf_root = tmp_path / "PEMF"
    source = _video(
        pemf_root,
        "video/Original/S001/Neutral/Neutral S001.mp4",
    )

    result = prepare_pemf.prepare_dataset(
        pemf_root,
        metadata_path=WORKBOOK_PATH,
        validate_expected_counts=False,
        dry_run=True,
    )

    assert result["sample_count"] == 1
    assert result["moved_count"] == 1
    assert source.is_file()
    assert not (pemf_root / "video" / "Original" / "S001" / "S001N.mp4").exists()
    assert not (pemf_root / "starting_point" / "samples.csv").exists()


def test_csv_only_writes_float_labels_without_moving_videos(tmp_path):
    prepare_pemf = _load_prepare_pemf()
    pemf_root = tmp_path / "PEMF"
    source = _video(
        pemf_root,
        "video/Original/S001/Algometer Pain/Algometer Pain S001.mp4",
    )

    result = prepare_pemf.prepare_dataset(
        pemf_root,
        metadata_path=WORKBOOK_PATH,
        validate_expected_counts=False,
        csv_only=True,
        class_id_type="float",
    )

    assert result["moved_count"] == 0
    assert source.is_file()
    assert not (pemf_root / "video" / "Original" / "S001" / "S001A.mp4").exists()
    columns, rows = _read_tsv(pemf_root / "starting_point" / "samples.csv")
    assert columns == EXPECTED_COLUMNS
    assert rows[0]["sample_name"] == "S001A"
    assert rows[0]["class_id"] == "5.20"


def test_csv_only_dry_run_writes_nothing(tmp_path):
    prepare_pemf = _load_prepare_pemf()
    pemf_root = tmp_path / "PEMF"
    source = _video(
        pemf_root,
        "video/Original/S001/Neutral/Neutral S001.mp4",
    )

    result = prepare_pemf.prepare_dataset(
        pemf_root,
        metadata_path=WORKBOOK_PATH,
        validate_expected_counts=False,
        csv_only=True,
        dry_run=True,
    )

    assert result["moved_count"] == 1
    assert source.is_file()
    assert not (pemf_root / "starting_point" / "samples.csv").exists()


def test_cli_accepts_csv_only_and_class_id_type():
    prepare_pemf = _load_prepare_pemf()

    args = prepare_pemf._parse_args(
        ["--csv-only", "--class-id-type", "float"]
    )

    assert args.csv_only is True
    assert args.class_id_type == "float"


def test_conflict_is_rejected_before_any_video_is_moved(tmp_path):
    prepare_pemf = _load_prepare_pemf()
    pemf_root = tmp_path / "PEMF"
    original = pemf_root / "video" / "Original"
    safe_source = _video(original, "S001/Neutral/Neutral S001.mp4")
    conflicting_source = _video(original, "S002/Neutral/Neutral S002.mp4")
    conflicting_target = _video(original, "S002/S002N.mp4", b"keep")

    with pytest.raises(ValueError, match="Refusing to overwrite"):
        prepare_pemf.prepare_dataset(
            pemf_root,
            metadata_path=WORKBOOK_PATH,
            validate_expected_counts=False,
        )

    assert safe_source.is_file()
    assert conflicting_source.is_file()
    assert conflicting_target.read_bytes() == b"keep"
    assert not (original / "S001" / "S001N.mp4").exists()
    assert not (pemf_root / "starting_point" / "samples.csv").exists()


def test_move_refuses_destination_created_after_preflight(tmp_path):
    prepare_pemf = _load_prepare_pemf()
    source = _video(tmp_path, "source.mp4", b"source")
    destination = _video(tmp_path, "destination.mp4", b"destination")

    with pytest.raises(ValueError, match="Refusing to overwrite"):
        prepare_pemf._move_videos([(source, destination)])

    assert source.read_bytes() == b"source"
    assert destination.read_bytes() == b"destination"


def test_prepare_dataset_is_idempotent(tmp_path):
    prepare_pemf = _load_prepare_pemf()
    pemf_root = tmp_path / "PEMF"
    _video(
        pemf_root,
        "video/Original/S001/Posed Pain/S001F.mp4",
    )

    first = prepare_pemf.prepare_dataset(
        pemf_root,
        metadata_path=WORKBOOK_PATH,
        validate_expected_counts=False,
    )
    csv_path = pemf_root / "starting_point" / "samples.csv"
    first_csv = csv_path.read_bytes()
    second = prepare_pemf.prepare_dataset(
        pemf_root,
        metadata_path=WORKBOOK_PATH,
        validate_expected_counts=False,
    )

    assert first["moved_count"] == 1
    assert second["moved_count"] == 0
    assert second["sample_count"] == 1
    assert csv_path.read_bytes() == first_csv
    assert (pemf_root / "video" / "Original" / "S001" / "S001P.mp4").is_file()
