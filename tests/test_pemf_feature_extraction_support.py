import os
from pathlib import Path
import subprocess
import sys


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


def test_pemf_paths_select_non_overlapping_augmentation_ids():
    helper.set_step_shift("PEMF/video/video_frontalized")

    assert helper.step_shift == 277
    assert helper.transform_sample_id(277, "hflip") == 554


def test_batch_script_routes_pemf_to_spatial_model_features(tmp_path):
    environment, call_log = _install_python_stub(tmp_path)

    result = subprocess.run(
        [
            "bash",
            str(SCRIPT_PATH),
            "0",
            "1",
            "--dataset",
            "pemf",
            "--model",
            "DFER",
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
    assert len(calls) == 1
    assert calls[0].startswith("extract_feature.py ")
    assert "--model_type DFER --emb_red spatial" in calls[0]
    assert "--path_dataset PEMF/video/video_frontalized" in calls[0]
    assert "--path_labels PEMF/starting_point/samples.csv" in calls[0]
    assert (
        "--saving_folder_path "
        "PEMF/video/features/DFER/"
        "spatial_pooled_features_PEMF_B_last143_stride16_interpol_all_shift"
    ) in calls[0]
