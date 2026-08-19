import csv
import importlib.util
from pathlib import Path

import cv2
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPOSITORY_ROOT / "XITE" / "check_frontalized_videos.py"
CSV_COLUMNS = [
    "subject_id",
    "subject_name",
    "class_id",
    "class_name",
    "sample_id",
    "sample_name",
]


def _load_checker():
    assert SCRIPT_PATH.is_file(), f"Missing checker script: {SCRIPT_PATH}"
    spec = importlib.util.spec_from_file_location(
        "check_xite_frontalized_videos", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_video(path: Path, *, frames: int = 16, fps: float = 25.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (32, 24)
    )
    assert writer.isOpened()
    for index in range(frames):
        frame = np.full((24, 32, 3), index % 256, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def _issue_kinds(issues):
    return [issue.kind for issue in issues]


def _write_labels(path: Path, rows) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def test_issue_csvs_split_warning_and_error_rows_in_source_order(tmp_path):
    checker = _load_checker()
    labels = tmp_path / "samples.csv"
    warning_csv = tmp_path / "warning_samples.csv"
    error_csv = tmp_path / "error_samples.csv"
    _write_labels(
        labels,
        [
            {
                "subject_id": "1",
                "subject_name": "S001",
                "class_id": "0",
                "class_name": "PL1",
                "sample_id": "10",
                "sample_name": "S001_2",
            },
            {
                "subject_id": "1",
                "subject_name": "S001",
                "class_id": "1",
                "class_name": "PL2",
                "sample_id": "8",
                "sample_name": "S001_0",
            },
        ],
    )
    result = checker.AuditResult(
        original_count=2,
        frontalized_count=1,
        errors=[
            checker.Issue(
                "missing_frontalized",
                "S001/S001_2.mp4",
                "original exists but frontalized output is missing",
            ),
            checker.Issue(
                "too_short", "S001/S001_0.mp4", "8 frames; requires at least 16"
            ),
        ],
        warnings=[
            checker.Issue(
                "frame_loss",
                "S001/S001_0.mp4",
                "32 original frames, 8 frontalized frames",
            )
        ],
    )

    warning_count, error_count = checker.write_issue_csvs(
        labels, warning_csv, error_csv, result
    )

    with warning_csv.open(newline="", encoding="utf-8") as handle:
        warning_rows = list(csv.DictReader(handle, delimiter="\t"))
    with error_csv.open(newline="", encoding="utf-8") as handle:
        error_rows = list(csv.DictReader(handle, delimiter="\t"))
    assert (warning_count, error_count) == (1, 2)
    assert warning_rows == [
        {
            "subject_id": "1",
            "subject_name": "S001",
            "class_id": "1",
            "class_name": "PL2",
            "sample_id": "8",
            "sample_name": "S001_0",
        }
    ]
    assert error_rows == [
        {
            "subject_id": "1",
            "subject_name": "S001",
            "class_id": "0",
            "class_name": "PL1",
            "sample_id": "10",
            "sample_name": "S001_2",
        },
        {
            "subject_id": "1",
            "subject_name": "S001",
            "class_id": "1",
            "class_name": "PL2",
            "sample_id": "8",
            "sample_name": "S001_0",
        },
    ]


def test_repair_csv_excludes_original_and_orphan_output_issues(tmp_path):
    checker = _load_checker()
    labels = tmp_path / "samples.csv"
    repair_csv = tmp_path / "repair_samples.csv"
    _write_labels(
        labels,
        [
            {
                "subject_id": "1",
                "subject_name": "S001",
                "class_id": "0",
                "class_name": "PL1",
                "sample_id": "1",
                "sample_name": "S001_0",
            },
            {
                "subject_id": "2",
                "subject_name": "S002",
                "class_id": "1",
                "class_name": "PL2",
                "sample_id": "2",
                "sample_name": "S002_0",
            },
        ],
    )
    result = checker.AuditResult(
        original_count=1,
        frontalized_count=1,
        errors=[
            checker.Issue("unreadable_original", "S001/S001_0.mp4", "bad input"),
            checker.Issue("unexpected_frontalized", "S002/S002_0.mp4", "orphan"),
        ],
        warnings=[],
    )

    written_count = checker.write_repair_csv(labels, repair_csv, result)

    with repair_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert written_count == 0
    assert rows == []


def test_clean_audit_replaces_stale_repair_csv_with_header_only_file(tmp_path):
    checker = _load_checker()
    labels = tmp_path / "samples.csv"
    repair_csv = tmp_path / "repair_samples.csv"
    _write_labels(
        labels,
        [
            {
                "subject_id": "1",
                "subject_name": "S001",
                "class_id": "0",
                "class_name": "PL1",
                "sample_id": "1",
                "sample_name": "S001_0",
            }
        ],
    )
    repair_csv.write_text("stale data\n", encoding="utf-8")
    result = checker.AuditResult(1, 1, errors=[], warnings=[])

    written_count = checker.write_repair_csv(labels, repair_csv, result)

    assert written_count == 0
    assert repair_csv.read_text(encoding="utf-8") == "\t".join(CSV_COLUMNS) + "\n"


def test_repair_csv_rejects_noncanonical_metadata_without_replacing_output(tmp_path):
    checker = _load_checker()
    labels = tmp_path / "samples.csv"
    labels.write_text("subject_name\tsample_name\nS001\tS001_0\n", encoding="utf-8")
    repair_csv = tmp_path / "repair_samples.csv"
    repair_csv.write_text("existing repair data\n", encoding="utf-8")
    result = checker.AuditResult(
        1,
        0,
        errors=[checker.Issue("missing_frontalized", "S001/S001_0.mp4", "missing")],
        warnings=[],
    )

    try:
        checker.write_repair_csv(labels, repair_csv, result)
    except ValueError as error:
        assert "must contain columns" in str(error)
        assert "subject_id" in str(error)
    else:
        raise AssertionError("Expected noncanonical metadata to be rejected")

    assert repair_csv.read_text(encoding="utf-8") == "existing repair data\n"


def test_repair_csv_rejects_duplicate_metadata_paths_without_replacing_output(
    tmp_path,
):
    checker = _load_checker()
    labels = tmp_path / "samples.csv"
    repair_csv = tmp_path / "repair_samples.csv"
    _write_labels(
        labels,
        [
            {
                "subject_id": "1",
                "subject_name": "S001",
                "class_id": "0",
                "class_name": "PL1",
                "sample_id": "1",
                "sample_name": "S001_0",
            },
            {
                "subject_id": "1",
                "subject_name": "S001",
                "class_id": "0",
                "class_name": "PL1",
                "sample_id": "2",
                "sample_name": "S001_0",
            },
        ],
    )
    repair_csv.write_text("existing repair data\n", encoding="utf-8")
    result = checker.AuditResult(
        1,
        0,
        errors=[checker.Issue("missing_frontalized", "S001/S001_0.mp4", "missing")],
        warnings=[],
    )

    try:
        checker.write_repair_csv(labels, repair_csv, result)
    except ValueError as error:
        assert "Duplicate metadata sample path: S001/S001_0.mp4" in str(error)
    else:
        raise AssertionError("Expected duplicate metadata paths to be rejected")

    assert repair_csv.read_text(encoding="utf-8") == "existing repair data\n"


def test_repair_csv_rejects_unmapped_repair_paths_without_replacing_output(tmp_path):
    checker = _load_checker()
    labels = tmp_path / "samples.csv"
    repair_csv = tmp_path / "repair_samples.csv"
    _write_labels(
        labels,
        [
            {
                "subject_id": "1",
                "subject_name": "S001",
                "class_id": "0",
                "class_name": "PL1",
                "sample_id": "1",
                "sample_name": "S001_0",
            }
        ],
    )
    repair_csv.write_text("existing repair data\n", encoding="utf-8")
    result = checker.AuditResult(
        1,
        0,
        errors=[checker.Issue("missing_frontalized", "S002/S002_0.mp4", "missing")],
        warnings=[],
    )

    try:
        checker.write_repair_csv(labels, repair_csv, result)
    except ValueError as error:
        assert "Repairable audit samples missing from metadata" in str(error)
        assert "S002/S002_0.mp4" in str(error)
    else:
        raise AssertionError("Expected unmapped repair paths to be rejected")

    assert repair_csv.read_text(encoding="utf-8") == "existing repair data\n"


def test_issue_csvs_preserve_both_outputs_when_one_category_is_unmapped(tmp_path):
    checker = _load_checker()
    labels = tmp_path / "samples.csv"
    warning_csv = tmp_path / "warning_samples.csv"
    error_csv = tmp_path / "error_samples.csv"
    _write_labels(
        labels,
        [
            {
                "subject_id": "1",
                "subject_name": "S001",
                "class_id": "0",
                "class_name": "PL1",
                "sample_id": "1",
                "sample_name": "S001_0",
            }
        ],
    )
    warning_csv.write_text("existing warning data\n", encoding="utf-8")
    error_csv.write_text("existing error data\n", encoding="utf-8")
    result = checker.AuditResult(
        1,
        0,
        errors=[checker.Issue("missing_frontalized", "S002/S002_0.mp4", "missing")],
        warnings=[checker.Issue("frame_loss", "S001/S001_0.mp4", "frame loss")],
    )

    try:
        checker.write_issue_csvs(labels, warning_csv, error_csv, result)
    except ValueError as error:
        assert "Repairable audit samples missing from metadata" in str(error)
        assert "S002/S002_0.mp4" in str(error)
    else:
        raise AssertionError("Expected unmapped error paths to be rejected")

    assert warning_csv.read_text(encoding="utf-8") == "existing warning data\n"
    assert error_csv.read_text(encoding="utf-8") == "existing error data\n"


def test_audit_accepts_matching_readable_clip_aligned_videos(tmp_path):
    checker = _load_checker()
    original = tmp_path / "video"
    frontalized = tmp_path / "video_frontalized"
    _write_video(original / "S001" / "S001_0.mp4")
    _write_video(frontalized / "S001" / "S001_0.mp4")

    result = checker.audit_videos(original, frontalized, clip_length=16)

    assert result.original_count == 1
    assert result.frontalized_count == 1
    assert result.errors == []
    assert result.warnings == []


def test_audit_shows_progress_for_paired_videos(tmp_path, monkeypatch):
    checker = _load_checker()
    original = tmp_path / "video"
    frontalized = tmp_path / "video_frontalized"
    for sample_name in ("S001_0", "S001_1"):
        for root in (original, frontalized):
            path = root / "S001" / f"{sample_name}.mp4"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()

    progress_calls = []

    def fake_tqdm(iterable, **kwargs):
        progress_calls.append(kwargs)
        return iterable

    metadata = checker.VideoMetadata(True, 16, 25.0, 32, 24)
    monkeypatch.setattr(checker, "tqdm", fake_tqdm, raising=False)
    monkeypatch.setattr(checker, "_probe_video", lambda path: metadata)

    result = checker.audit_videos(
        original, frontalized, clip_length=16, show_progress=True
    )

    assert result.errors == []
    assert progress_calls == [
        {"desc": "Checking frontalized videos", "total": 2, "unit": "video"}
    ]


def test_audit_reports_missing_and_unexpected_outputs(tmp_path):
    checker = _load_checker()
    original = tmp_path / "video"
    frontalized = tmp_path / "video_frontalized"
    _write_video(original / "S001" / "S001_0.mp4")
    _write_video(frontalized / "S002" / "S002_0.mp4")

    result = checker.audit_videos(original, frontalized, clip_length=16)

    assert _issue_kinds(result.errors) == [
        "missing_frontalized",
        "unexpected_frontalized",
    ]


def test_audit_warns_without_failing_when_frontalization_loses_frames(tmp_path):
    checker = _load_checker()
    original = tmp_path / "video"
    frontalized = tmp_path / "video_frontalized"
    _write_video(original / "S001" / "S001_0.mp4", frames=32)
    _write_video(frontalized / "S001" / "S001_0.mp4", frames=16)

    result = checker.audit_videos(original, frontalized, clip_length=16)

    assert result.errors == []
    assert _issue_kinds(result.warnings) == ["frame_loss"]
    assert "32 original frames, 16 frontalized frames" in result.warnings[0].detail


def test_audit_rejects_short_and_non_aligned_outputs(tmp_path):
    checker = _load_checker()
    original = tmp_path / "video"
    frontalized = tmp_path / "video_frontalized"
    _write_video(original / "S001" / "S001_0.mp4", frames=32)
    _write_video(original / "S001" / "S001_1.mp4", frames=32)
    _write_video(frontalized / "S001" / "S001_0.mp4", frames=8)
    _write_video(frontalized / "S001" / "S001_1.mp4", frames=17)

    result = checker.audit_videos(original, frontalized, clip_length=16)

    assert _issue_kinds(result.errors) == ["too_short", "not_clip_aligned"]
    assert _issue_kinds(result.warnings) == ["frame_loss", "frame_loss"]


def test_audit_rejects_unreadable_output_and_fps_mismatch(tmp_path):
    checker = _load_checker()
    original = tmp_path / "video"
    frontalized = tmp_path / "video_frontalized"
    _write_video(original / "S001" / "S001_0.mp4")
    _write_video(original / "S001" / "S001_1.mp4")
    bad_output = frontalized / "S001" / "S001_0.mp4"
    bad_output.parent.mkdir(parents=True, exist_ok=True)
    bad_output.touch()
    _write_video(frontalized / "S001" / "S001_1.mp4", fps=20.0)

    result = checker.audit_videos(original, frontalized, clip_length=16)

    assert _issue_kinds(result.errors) == ["unreadable_frontalized", "fps_mismatch"]


def test_audit_rejects_non_positive_and_non_finite_fps(tmp_path, monkeypatch):
    checker = _load_checker()
    original = tmp_path / "video"
    frontalized = tmp_path / "video_frontalized"
    for sample_name in ("S001_0", "S001_1"):
        for root in (original, frontalized):
            path = root / "S001" / f"{sample_name}.mp4"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()

    def probe(path):
        is_frontalized = path.parents[1].name == "video_frontalized"
        if path.stem == "S001_0":
            fps = 25.0 if is_frontalized else 0.0
        else:
            fps = float("nan") if is_frontalized else 25.0
        return checker.VideoMetadata(
            readable=True,
            frame_count=16,
            fps=fps,
            width=32,
            height=24,
        )

    monkeypatch.setattr(checker, "_probe_video", probe)

    result = checker.audit_videos(original, frontalized, clip_length=16)

    assert _issue_kinds(result.errors) == [
        "invalid_original_metadata",
        "invalid_frontalized_metadata",
    ]


def test_labels_filter_limits_the_expected_inventory(tmp_path):
    checker = _load_checker()
    original = tmp_path / "video"
    frontalized = tmp_path / "video_frontalized"
    _write_video(original / "S001" / "S001_0.mp4")
    _write_video(original / "S002" / "S002_0.mp4")
    _write_video(frontalized / "S001" / "S001_0.mp4")
    labels = tmp_path / "test_samples.csv"
    with labels.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["subject_name", "sample_name"],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerow({"subject_name": "S001", "sample_name": "S001_0"})

    result = checker.audit_videos(
        original, frontalized, clip_length=16, labels_path=labels
    )

    assert result.original_count == 1
    assert result.frontalized_count == 1
    assert result.errors == []


def test_cli_returns_failure_and_prints_summary_for_hard_errors(
    tmp_path, capsys, monkeypatch
):
    checker = _load_checker()
    original = tmp_path / "video"
    frontalized = tmp_path / "video_frontalized"
    _write_video(original / "S001" / "S001_0.mp4")
    frontalized.mkdir()
    labels = tmp_path / "samples.csv"
    warning_csv = tmp_path / "warning_samples.csv"
    error_csv = tmp_path / "error_samples.csv"
    _write_labels(
        labels,
        [
            {
                "subject_id": "1",
                "subject_name": "S001",
                "class_id": "0",
                "class_name": "PL1",
                "sample_id": "1",
                "sample_name": "S001_0",
            }
        ],
    )
    monkeypatch.setattr(checker, "DEFAULT_METADATA_PATH", labels)
    monkeypatch.setattr(checker, "DEFAULT_WARNING_CSV_PATH", warning_csv)
    monkeypatch.setattr(checker, "DEFAULT_ERROR_CSV_PATH", error_csv)

    exit_code = checker.main(
        [
            "--original-root",
            str(original),
            "--frontalized-root",
            str(frontalized),
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 1
    assert "ERROR [missing_frontalized] S001/S001_0.mp4" in output
    assert "originals=1 frontalized=0 errors=1 warnings=0" in output
    assert f"Error CSV: {error_csv} samples=1" in output
    assert f"Warning CSV: {warning_csv} samples=0" in output
    with error_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert [row["sample_name"] for row in rows] == ["S001_0"]
    with warning_csv.open(newline="", encoding="utf-8") as handle:
        assert list(csv.DictReader(handle, delimiter="\t")) == []


def test_cli_writes_warning_samples_without_changing_success_exit_code(
    tmp_path, capsys, monkeypatch
):
    checker = _load_checker()
    original = tmp_path / "video"
    frontalized = tmp_path / "video_frontalized"
    _write_video(original / "S001" / "S001_0.mp4", frames=32)
    _write_video(frontalized / "S001" / "S001_0.mp4", frames=16)
    labels = tmp_path / "samples.csv"
    warning_csv = tmp_path / "warning_samples.csv"
    error_csv = tmp_path / "error_samples.csv"
    _write_labels(
        labels,
        [
            {
                "subject_id": "1",
                "subject_name": "S001",
                "class_id": "0",
                "class_name": "PL1",
                "sample_id": "1",
                "sample_name": "S001_0",
            }
        ],
    )
    monkeypatch.setattr(checker, "DEFAULT_METADATA_PATH", labels)
    monkeypatch.setattr(checker, "DEFAULT_WARNING_CSV_PATH", warning_csv)
    monkeypatch.setattr(checker, "DEFAULT_ERROR_CSV_PATH", error_csv)

    exit_code = checker.main(
        [
            "--original-root",
            str(original),
            "--frontalized-root",
            str(frontalized),
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "WARNING [frame_loss] S001/S001_0.mp4" in output
    assert f"Warning CSV: {warning_csv} samples=1" in output
    assert f"Error CSV: {error_csv} samples=0" in output
    with warning_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert [row["sample_name"] for row in rows] == ["S001_0"]
