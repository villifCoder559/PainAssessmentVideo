import os
import subprocess
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from custom import helper


SCRIPT_PATH = REPOSITORY_ROOT / "multiple_feature_extraction.sh"


def _install_python_stub(tmp_path: Path) -> tuple[dict, Path]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    call_log = tmp_path / "python_calls.txt"
    stub = bin_dir / "python3"
    stub.write_text(
        "#!/bin/sh\nprintf '%s\\n' \"$*\" >> \"$CALL_LOG\"\n",
        encoding="utf-8",
    )
    stub.chmod(0o755)
    environment = os.environ.copy()
    environment["PATH"] = f"{bin_dir}:{environment['PATH']}"
    environment["CALL_LOG"] = str(call_log)
    return environment, call_log


def test_xite_paths_select_global_non_overlapping_augmentation_ids():
    helper.set_step_shift("XITE/video/video_frontalized")

    assert helper.step_shift == 3597
    assert helper.transform_sample_id(3597, "hflip") == 7194


def test_batch_script_selects_xite_test_metadata_and_preflights_it(tmp_path):
    environment, call_log = _install_python_stub(tmp_path)

    result = subprocess.run(
        [
            "bash",
            str(SCRIPT_PATH),
            "0",
            "1",
            "--dataset",
            "xite",
            "--split",
            "test",
            "--model",
            "S",
            "shift",
        ],
        cwd=REPOSITORY_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr + result.stdout
    calls = call_log.read_text(encoding="utf-8").splitlines()
    assert calls[0] == (
        "XITE/check_frontalized_videos.py "
        "--labels XITE/starting_point/test_samples.csv --clip-length 16"
    )
    assert calls[1].startswith("extract_feature.py ")
    assert "--path_dataset XITE/video/video_frontalized" in calls[1]
    assert "--path_labels XITE/starting_point/test_samples.csv" in calls[1]
    assert (
        "--saving_folder_path "
        "XITE/video/features/VideoMaev2_S/"
        "spatial_pooled_features_XITE_B_last143_stride16_interpol_test_shift"
    ) in calls[1]


def test_batch_script_defaults_xite_to_training_split(tmp_path):
    environment, call_log = _install_python_stub(tmp_path)

    result = subprocess.run(
        [
            "bash",
            str(SCRIPT_PATH),
            "0",
            "1",
            "--dataset",
            "xite",
            "--model",
            "S",
            "jitter",
        ],
        cwd=REPOSITORY_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr + result.stdout
    calls = call_log.read_text(encoding="utf-8").splitlines()
    assert "--labels XITE/starting_point/train_samples.csv" in calls[0]
    assert "--path_labels XITE/starting_point/train_samples.csv" in calls[1]
    assert "stride16_interpol_train_jitter" in calls[1]


def test_batch_script_rejects_split_for_non_xite_dataset(tmp_path):
    environment, call_log = _install_python_stub(tmp_path)

    result = subprocess.run(
        [
            "bash",
            str(SCRIPT_PATH),
            "0",
            "1",
            "--dataset",
            "biovid",
            "--split",
            "test",
            "shift",
        ],
        cwd=REPOSITORY_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    assert "--split is only valid with --dataset xite" in result.stderr
    assert not call_log.exists()
