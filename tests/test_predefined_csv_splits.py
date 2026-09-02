from pathlib import Path
import sys

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


REQUIRED_COLUMNS = (
    "subject_id",
    "subject_name",
    "class_id",
    "class_name",
    "sample_id",
    "sample_name",
)


def _write_split(path: Path, rows):
    header = "\t".join(REQUIRED_COLUMNS)
    body = "\n".join("\t".join(str(value) for value in row) for row in rows)
    path.write_text(f"{header}\n{body}\n", encoding="utf-8")


def _valid_rows(subject_id, subject_name, sample_id):
    return [(subject_id, subject_name, 0, "PL1", sample_id, f"sample_{sample_id}")]


def test_csv_file_keeps_legacy_mode(tmp_path):
    from custom.predefined_splits import discover_predefined_csv_splits

    csv_path = tmp_path / "dataset.csv"
    _write_split(csv_path, _valid_rows(1, "S001", 1))

    assert discover_predefined_csv_splits(csv_path, subject_independent=True) is None


def test_directory_maps_three_csvs_by_case_insensitive_substring(tmp_path):
    from custom.predefined_splits import discover_predefined_csv_splits

    train_path = tmp_path / "myTRAINing.csv"
    val_path = tmp_path / "evaluation.csv"
    test_path = tmp_path / "held_TEST_set.CSV"
    _write_split(train_path, _valid_rows(1, "S001", 1))
    _write_split(val_path, _valid_rows(2, "S002", 2))
    _write_split(test_path, _valid_rows(3, "S003", 3))

    assert discover_predefined_csv_splits(
        tmp_path, subject_independent=True
    ) == {
        "train": str(train_path),
        "val": str(val_path),
        "test": str(test_path),
    }


def test_directory_requires_exactly_three_immediate_csvs(tmp_path):
    from custom.predefined_splits import discover_predefined_csv_splits

    _write_split(tmp_path / "train.csv", _valid_rows(1, "S001", 1))
    _write_split(tmp_path / "val.csv", _valid_rows(2, "S002", 2))
    nested = tmp_path / "nested"
    nested.mkdir()
    _write_split(nested / "test.csv", _valid_rows(3, "S003", 3))

    with pytest.raises(ValueError, match="exactly 3 immediate CSV files"):
        discover_predefined_csv_splits(tmp_path, subject_independent=True)


def test_directory_rejects_filename_matching_multiple_roles(tmp_path):
    from custom.predefined_splits import discover_predefined_csv_splits

    _write_split(tmp_path / "train_val.csv", _valid_rows(1, "S001", 1))
    _write_split(tmp_path / "val_extra.csv", _valid_rows(2, "S002", 2))
    _write_split(tmp_path / "test.csv", _valid_rows(3, "S003", 3))

    with pytest.raises(ValueError, match="matches multiple split roles"):
        discover_predefined_csv_splits(tmp_path, subject_independent=True)


def test_directory_rejects_missing_required_columns(tmp_path):
    from custom.predefined_splits import discover_predefined_csv_splits

    _write_split(tmp_path / "train.csv", _valid_rows(1, "S001", 1))
    _write_split(tmp_path / "val.csv", _valid_rows(2, "S002", 2))
    (tmp_path / "test.csv").write_text(
        "subject_id\tsubject_name\n3\tS003\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="missing required columns"):
        discover_predefined_csv_splits(tmp_path, subject_independent=True)


def test_directory_rejects_empty_split(tmp_path):
    from custom.predefined_splits import discover_predefined_csv_splits

    _write_split(tmp_path / "train.csv", _valid_rows(1, "S001", 1))
    _write_split(tmp_path / "val.csv", [])
    _write_split(tmp_path / "test.csv", _valid_rows(3, "S003", 3))

    with pytest.raises(ValueError, match="must contain at least one row"):
        discover_predefined_csv_splits(tmp_path, subject_independent=True)


def test_subject_overlap_is_rejected_only_when_requested(tmp_path):
    from custom.predefined_splits import discover_predefined_csv_splits

    _write_split(tmp_path / "train.csv", _valid_rows(7, "TRAIN_NAME", 1))
    _write_split(tmp_path / "val.csv", _valid_rows(7, "OTHER_NAME", 2))
    _write_split(tmp_path / "test.csv", _valid_rows(9, "S009", 3))

    with pytest.raises(ValueError, match=r"subject_id.*train.*val.*7"):
        discover_predefined_csv_splits(tmp_path, subject_independent=True)

    assert discover_predefined_csv_splits(
        tmp_path, subject_independent=False
    )["train"].endswith("train.csv")


def test_configure_csv_input_keeps_legacy_file_settings(tmp_path):
    from custom.predefined_splits import configure_csv_input

    csv_path = tmp_path / "dataset.csv"
    _write_split(csv_path, _valid_rows(1, "S001", 1))
    config = {
        "csv": str(csv_path),
        "is_subject_independent": 1,
        "validation_enabled": 1,
        "skip_test": 0,
        "use_test_as_val": 0,
        "stop": [2, 3],
    }

    configure_csv_input(config)

    assert config["training_csv"] == str(csv_path)
    assert config["predefined_csv_splits"] is None
    assert config["stop"] == [2, 3]


def test_configure_csv_input_forces_single_run_for_directory(tmp_path):
    from custom.predefined_splits import configure_csv_input

    _write_split(tmp_path / "train.csv", _valid_rows(1, "S001", 1))
    _write_split(tmp_path / "val.csv", _valid_rows(2, "S002", 2))
    _write_split(tmp_path / "test.csv", _valid_rows(3, "S003", 3))
    config = {
        "csv": str(tmp_path),
        "is_subject_independent": 1,
        "validation_enabled": 1,
        "skip_test": 0,
        "use_test_as_val": 0,
        "stop": [4, 5],
    }

    configure_csv_input(config)

    assert config["training_csv"].endswith("train.csv")
    assert config["predefined_csv_splits"]["test"].endswith("test.csv")
    assert config["stop"] == [1, 1]


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"validation_enabled": 0}, "validation_enabled"),
        ({"skip_test": 1}, "skip_test"),
        ({"use_test_as_val": 1}, "use_test_as_val"),
    ],
)
def test_configure_csv_input_rejects_flags_that_change_split_roles(
    tmp_path, override, message
):
    from custom.predefined_splits import configure_csv_input

    _write_split(tmp_path / "train.csv", _valid_rows(1, "S001", 1))
    _write_split(tmp_path / "val.csv", _valid_rows(2, "S002", 2))
    _write_split(tmp_path / "test.csv", _valid_rows(3, "S003", 3))
    config = {
        "csv": str(tmp_path),
        "is_subject_independent": 1,
        "validation_enabled": 1,
        "skip_test": 0,
        "use_test_as_val": 0,
        "stop": None,
        **override,
    }

    with pytest.raises(ValueError, match=message):
        configure_csv_input(config)
