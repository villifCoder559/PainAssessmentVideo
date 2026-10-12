import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = REPO_ROOT / "run_cross_space_anchor_sweep.sh"


def test_runs_each_distinct_anchor_with_an_isolated_config_and_reports_failures(tmp_path):
    sandbox = tmp_path / "repo"
    sandbox.mkdir()
    launcher = sandbox / LAUNCHER.name
    shutil.copy2(LAUNCHER, launcher)

    call_log = sandbox / "calls.jsonl"
    projection_stub = sandbox / "cross_space_projection.py"
    projection_stub.write_text(
        """
import json
import os
import sys

import yaml

config_path = sys.argv[sys.argv.index("--config") + 1]
with open(config_path, encoding="utf-8") as stream:
    config = yaml.safe_load(stream)
with open(os.environ["CALL_LOG"], "a", encoding="utf-8") as stream:
    stream.write(json.dumps({
        "config": config,
        "config_path": config_path,
        "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }) + "\\n")
if config["num_anchors"] == [50]:
    raise SystemExit(7)
""".lstrip(),
        encoding="utf-8",
    )

    source_config = sandbox / "experiment.yaml"
    source = {
        "new_model_pth": ["new-fold-0.pt", "new-fold-1.pt"],
        "old_model_pth": ["old-fold-0.pt", "old-fold-1.pt"],
        "num_anchors": [100, 50, 100, 250],
        "run_tag": "group/base",
        "linear_projector": {"epochs": 17, "loss": ["mse"]},
    }
    source_config.write_text(yaml.safe_dump(source, sort_keys=False), encoding="utf-8")
    source_before = source_config.read_bytes()

    environment = os.environ.copy()
    environment["CALL_LOG"] = str(call_log)
    result = subprocess.run(
        ["bash", str(launcher), "3", str(source_config)],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1, result.stdout + result.stderr
    calls = [json.loads(line) for line in call_log.read_text(encoding="utf-8").splitlines()]
    assert [call["config"]["num_anchors"] for call in calls] == [[100], [50], [250]]
    assert [call["config"]["run_tag"] for call in calls] == [
        "group/base/K100",
        "group/base/K50",
        "group/base/K250",
    ]
    assert all(call["gpu"] == "3" for call in calls)
    assert all(call["config"]["new_model_pth"] == source["new_model_pth"] for call in calls)
    assert all(call["config"]["old_model_pth"] == source["old_model_pth"] for call in calls)
    assert all(call["config"]["linear_projector"] == source["linear_projector"] for call in calls)
    assert all(not Path(call["config_path"]).exists() for call in calls)
    assert source_config.read_bytes() == source_before
    assert "Passed anchor values: 2" in result.stdout
    assert "Failed anchor values: 1" in result.stdout
    assert "Failed num_anchors values:\n  50\n" in result.stdout


def test_derives_run_tag_from_config_filename_when_missing(tmp_path):
    sandbox = tmp_path / "repo"
    sandbox.mkdir()
    launcher = sandbox / LAUNCHER.name
    shutil.copy2(LAUNCHER, launcher)

    call_log = sandbox / "calls.jsonl"
    (sandbox / "cross_space_projection.py").write_text(
        """
import json
import os
import sys

import yaml

with open(sys.argv[sys.argv.index("--config") + 1], encoding="utf-8") as stream:
    config = yaml.safe_load(stream)
with open(os.environ["CALL_LOG"], "w", encoding="utf-8") as stream:
    json.dump(config, stream)
""".lstrip(),
        encoding="utf-8",
    )
    source_config = sandbox / "experiment.yaml"
    source_config.write_text("num_anchors: [10]\n", encoding="utf-8")
    environment = os.environ.copy()
    environment["CALL_LOG"] = str(call_log)

    result = subprocess.run(
        ["bash", str(launcher), "0", str(source_config)],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    generated = json.loads(call_log.read_text(encoding="utf-8"))
    assert generated["run_tag"] == "experiment/K10"
    assert "Passed anchor values: 1" in result.stdout
    assert "Failed anchor values: 0" in result.stdout


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ([], "Usage:"),
        (["gpu", "unused.yaml"], "GPU_ID must be a non-negative integer"),
        (["0", "missing.yaml"], "Config YAML not found"),
    ],
)
def test_rejects_invalid_command_line_inputs(arguments, message):
    result = subprocess.run(
        ["bash", str(LAUNCHER), *arguments],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert message in result.stderr


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        ("[not, a, mapping]\n", "top-level mapping"),
        ("run_tag: experiment\n", "non-empty YAML list of integers"),
        ("num_anchors: []\n", "non-empty YAML list of integers"),
        ("num_anchors: [10, true]\n", "non-empty YAML list of integers"),
        ("num_anchors: [10]\nrun_tag: 42\n", "run_tag must be a string"),
        ("num_anchors: [10\n", "Invalid YAML"),
    ],
)
def test_rejects_invalid_yaml_configuration(tmp_path, contents, message):
    config = tmp_path / "invalid.yaml"
    config.write_text(contents, encoding="utf-8")

    result = subprocess.run(
        ["bash", str(LAUNCHER), "0", str(config)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert message in result.stderr
