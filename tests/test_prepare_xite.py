import csv
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPOSITORY_ROOT / "XITE" / "starting_point" / "prepare_xite.py"
EXPECTED_COLUMNS = [
    "subject_id",
    "subject_name",
    "class_id",
    "class_name",
    "sample_id",
    "sample_name",
]


def _load_prepare_xite():
    assert SCRIPT_PATH.is_file(), f"Missing preparation script: {SCRIPT_PATH}"
    spec = importlib.util.spec_from_file_location("prepare_xite", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _touch(root, relative_path):
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return path


def _miniature_xite_tree(tmp_path):
    root = tmp_path / "XITE"
    _touch(root, "train_data/low_pain/fvf/S010/S010_10.mp4")
    _touch(root, "train_data/low_pain/fvf/S010/S010_2.mp4")
    _touch(root, "train_data/med_pain/fvf/S010/S010_1.mp4")
    _touch(root, "train_data/med_pain/fvf/S020/S020_3.mp4")
    _touch(root, "test_data/fvf/S030 /S030_11.mp4")
    _touch(root, "test_data/fvf/S030 /S030_0.mp4")
    return root


def _challenge_sized_xite_tree(tmp_path):
    root = tmp_path / "XITE"
    for subject_number in range(1, 27):
        subject_name = f"S{subject_number:03d}"
        pl1_count = 59 if subject_number <= 3 else 60
        for segment in range(pl1_count):
            _touch(
                root,
                f"train_data/low_pain/fvf/{subject_name}/{subject_name}_{segment}.mp4",
            )
        for segment in range(60):
            _touch(
                root,
                f"train_data/med_pain/fvf/{subject_name}/{subject_name}_{segment + 100}.mp4",
            )
    for subject_number in range(1, 5):
        subject_name = f"T{subject_number:03d}"
        for segment in range(120):
            _touch(
                root,
                f"test_data/fvf/{subject_name}/{subject_name}_{segment}.mp4",
            )
    return root


def _read_tsv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return reader.fieldnames, list(reader)


def test_prepare_dataset_writes_part_a_compatible_csvs(tmp_path):
    prepare_xite = _load_prepare_xite()
    xite_root = _miniature_xite_tree(tmp_path)

    outputs = prepare_xite.prepare_dataset(
        xite_root,
        validate_expected_counts=False,
    )

    assert outputs == {
        "train": xite_root / "starting_point" / "train_samples.csv",
        "test": xite_root / "starting_point" / "test_samples.csv",
        "all": xite_root / "starting_point" / "samples.csv",
    }
    train_columns, train_rows = _read_tsv(outputs["train"])
    test_columns, test_rows = _read_tsv(outputs["test"])
    all_columns, all_rows = _read_tsv(outputs["all"])

    assert train_columns == test_columns == all_columns == EXPECTED_COLUMNS
    assert train_rows == [
        {
            "subject_id": "1",
            "subject_name": "S010",
            "class_id": "0",
            "class_name": "PL1",
            "sample_id": "1",
            "sample_name": "S010_2",
        },
        {
            "subject_id": "1",
            "subject_name": "S010",
            "class_id": "0",
            "class_name": "PL1",
            "sample_id": "2",
            "sample_name": "S010_10",
        },
        {
            "subject_id": "1",
            "subject_name": "S010",
            "class_id": "1",
            "class_name": "PL2",
            "sample_id": "3",
            "sample_name": "S010_1",
        },
        {
            "subject_id": "2",
            "subject_name": "S020",
            "class_id": "1",
            "class_name": "PL2",
            "sample_id": "4",
            "sample_name": "S020_3",
        },
    ]
    assert test_rows == [
        {
            "subject_id": "3",
            "subject_name": "S030",
            "class_id": "-1",
            "class_name": "UNKNOWN",
            "sample_id": "5",
            "sample_name": "S030_0",
        },
        {
            "subject_id": "3",
            "subject_name": "S030",
            "class_id": "-1",
            "class_name": "UNKNOWN",
            "sample_id": "6",
            "sample_name": "S030_11",
        },
    ]
    assert all_rows == train_rows + test_rows


def test_prepare_dataset_rejects_missing_fvf_source_directory(tmp_path):
    prepare_xite = _load_prepare_xite()
    xite_root = _miniature_xite_tree(tmp_path)
    missing_root = xite_root / "train_data" / "med_pain" / "fvf"
    shutil.rmtree(missing_root)

    with pytest.raises(ValueError, match=r"Missing required FVF directory: .*med_pain/fvf"):
        prepare_xite.prepare_dataset(
            xite_root,
            validate_expected_counts=False,
        )


def test_prepare_dataset_rejects_malformed_video_filename(tmp_path):
    prepare_xite = _load_prepare_xite()
    xite_root = _miniature_xite_tree(tmp_path)
    _touch(xite_root, "train_data/low_pain/fvf/S010/not-a-segment.mp4")

    with pytest.raises(
        ValueError,
        match=r"Expected '<subject>_<numeric-segment>\.mp4'.*not-a-segment\.mp4",
    ):
        prepare_xite.prepare_dataset(
            xite_root,
            validate_expected_counts=False,
        )


def test_prepare_dataset_rejects_duplicate_subject_sample_key(tmp_path):
    prepare_xite = _load_prepare_xite()
    xite_root = _miniature_xite_tree(tmp_path)
    _touch(xite_root, "train_data/med_pain/fvf/S010/S010_2.mp4")

    with pytest.raises(ValueError, match=r"Duplicate train sample key: S010/S010_2"):
        prepare_xite.prepare_dataset(
            xite_root,
            validate_expected_counts=False,
        )


def test_prepare_dataset_rejects_subject_overlap_between_splits(tmp_path):
    prepare_xite = _load_prepare_xite()
    xite_root = _miniature_xite_tree(tmp_path)
    _touch(xite_root, "test_data/fvf/S010/S010_99.mp4")

    with pytest.raises(ValueError, match=r"Subjects occur in both train and test: S010"):
        prepare_xite.prepare_dataset(
            xite_root,
            validate_expected_counts=False,
        )


def test_prepare_dataset_enforces_challenge_sample_counts_by_default(tmp_path):
    prepare_xite = _load_prepare_xite()
    xite_root = _miniature_xite_tree(tmp_path)

    with pytest.raises(
        ValueError,
        match=r"Unexpected PL1 sample count: expected 1557, found 2",
    ):
        prepare_xite.prepare_dataset(xite_root)


def test_failed_csv_regeneration_preserves_previous_file(tmp_path, monkeypatch):
    prepare_xite = _load_prepare_xite()
    xite_root = _miniature_xite_tree(tmp_path)
    outputs = prepare_xite.prepare_dataset(
        xite_root,
        validate_expected_counts=False,
    )
    previous_train_csv = outputs["train"].read_bytes()

    def fail_first_row(*args, **kwargs):
        raise OSError("simulated CSV write failure")

    monkeypatch.setattr(prepare_xite.csv.DictWriter, "writerow", fail_first_row)

    with pytest.raises(OSError, match="simulated CSV write failure"):
        prepare_xite.prepare_dataset(
            xite_root,
            validate_expected_counts=False,
        )

    assert outputs["train"].read_bytes() == previous_train_csv
    assert list(outputs["train"].parent.glob(".train_samples.csv.*.tmp")) == []


def test_cli_generates_validated_challenge_csvs_and_optional_video_links(tmp_path):
    xite_root = _challenge_sized_xite_tree(tmp_path)

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--xite-root",
            str(xite_root),
            "--reorganize-videos",
        ],
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Generated 3117 train, 480 test, and 3597 combined rows." in result.stdout
    assert "Prepared 3597 relative video symlinks" in result.stdout
    _, train_rows = _read_tsv(xite_root / "starting_point" / "train_samples.csv")
    _, test_rows = _read_tsv(xite_root / "starting_point" / "test_samples.csv")
    _, all_rows = _read_tsv(xite_root / "starting_point" / "samples.csv")
    assert (len(train_rows), len(test_rows), len(all_rows)) == (3117, 480, 3597)
    assert len(list((xite_root / "video" / "video").glob("*/*.mp4"))) == 3597


def test_reorganize_videos_creates_relative_part_a_compatible_symlinks(tmp_path):
    prepare_xite = _load_prepare_xite()
    xite_root = _miniature_xite_tree(tmp_path)

    prepare_xite.prepare_dataset(
        xite_root,
        validate_expected_counts=False,
        reorganize_videos=True,
    )

    video_root = xite_root / "video" / "video"
    links = sorted(video_root.glob("*/*.mp4"))
    assert len(links) == 6
    train_link = video_root / "S010" / "S010_2.mp4"
    test_link = video_root / "S030" / "S030_0.mp4"
    assert train_link.is_symlink()
    assert test_link.is_symlink()
    assert not os.path.isabs(os.readlink(train_link))
    assert not os.path.isabs(os.readlink(test_link))
    assert train_link.resolve() == (
        xite_root / "train_data" / "low_pain" / "fvf" / "S010" / "S010_2.mp4"
    ).resolve()
    assert test_link.resolve() == (
        xite_root / "test_data" / "fvf" / "S030 " / "S030_0.mp4"
    ).resolve()


def test_reorganize_videos_is_idempotent_for_correct_links(tmp_path):
    prepare_xite = _load_prepare_xite()
    xite_root = _miniature_xite_tree(tmp_path)
    outputs = prepare_xite.prepare_dataset(
        xite_root,
        validate_expected_counts=False,
        reorganize_videos=True,
    )
    csv_bytes = {name: path.read_bytes() for name, path in outputs.items()}

    rerun_outputs = prepare_xite.prepare_dataset(
        xite_root,
        validate_expected_counts=False,
        reorganize_videos=True,
    )

    assert rerun_outputs == outputs
    assert {name: path.read_bytes() for name, path in outputs.items()} == csv_bytes
    assert len(list((xite_root / "video" / "video").glob("*/*.mp4"))) == 6


def test_reorganize_videos_preflights_conflicts_before_creating_links(tmp_path):
    prepare_xite = _load_prepare_xite()
    xite_root = _miniature_xite_tree(tmp_path)
    video_root = xite_root / "video" / "video"
    conflicting_path = _touch(video_root, "S020/S020_3.mp4")

    with pytest.raises(
        ValueError,
        match=r"Refusing to overwrite conflicting video path: .*S020/S020_3\.mp4",
    ):
        prepare_xite.prepare_dataset(
            xite_root,
            validate_expected_counts=False,
            reorganize_videos=True,
        )

    assert conflicting_path.is_file()
    assert not (video_root / "S010" / "S010_2.mp4").exists()
