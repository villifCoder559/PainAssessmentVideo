import argparse
import copy
import os
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECTION_SCRIPT = REPO_ROOT / "cross_space_projection.py"
sys.path.insert(0, str(REPO_ROOT))


# custom.helper starts a multiprocessing Manager while cross_space_projection is
# imported. These tests stop before model processing and do not use that profiler.
with mock.patch("multiprocessing.Manager") as manager:
    manager.return_value.dict.return_value = {}
    import cross_space_projection as csp


RAW_CONFIG = (
    b"# generated anchor configuration\r\n"
    b"new_model_pth: /missing/new/a/b/c/model.pt\r\n"
    b"old_model_pth: /missing/old/a/b/c/model.pt\r\n"
    b"num_anchors: [10]  # preserve this comment\r\n"
    b"csv_anchor_selection: [train]\r\n"
    b"old_model_csv: [test]\r\n"
    b"run_tag: snapshots/standalone\r\n"
)


def _single_args(*, run_tag, launch_bytes=RAW_CONFIG):
    values = {
        "new_model_pth": "/missing/new/a/b/c/model.pt",
        "old_model_pth": "/missing/old/a/b/c/model.pt",
        "num_anchors": 10,
        "anchor_selection_type": "random",
        "csv_anchor_selection": "train",
        "old_model_csv": "test",
        "interpolation_similarity": "cos",
        "mlp_activation": "gelu",
        "mlp_num_layers": 1,
        "weighting_method": "rbf",
        "rbf_sigma": 1.0,
        "run_tag": run_tag,
        "fake_projection": False,
        "fake_projection_distribution": "matched_gaussian",
        "refinement": 0,
        "projector_recipes": [copy.deepcopy(csp.LINEAR_PROJECTOR_CONFIG)],
        "refinement_recipes": [copy.deepcopy(csp.REFINEMENT_CONFIG)],
    }
    if launch_bytes is not None:
        values["_launch_config_bytes"] = launch_bytes
    return argparse.Namespace(**values)


def _grid_args():
    args = _single_args(run_tag="snapshots/grid")
    for key in (
        "num_anchors",
        "anchor_selection_type",
        "csv_anchor_selection",
        "old_model_csv",
        "interpolation_similarity",
        "mlp_activation",
        "mlp_num_layers",
        "weighting_method",
        "rbf_sigma",
    ):
        setattr(args, key, [getattr(args, key)])
    args.n_trials = 1
    args.optuna_sampler = "grid"
    return args


class _CompletedStudy:
    def __init__(self):
        self.best_trial = SimpleNamespace(
            number=3,
            value=0.25,
            params={"num_anchors": 10},
        )

    def optimize(self, objective, n_trials):
        return None


def test_yaml_parser_preserves_the_exact_input_bytes_in_standalone_output(
    tmp_path, monkeypatch
):
    config = tmp_path / "generated.yaml"
    config.write_bytes(RAW_CONFIG)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", [str(PROJECTION_SCRIPT), "--config", str(config)])

    with pytest.raises(Exception):
        runpy.run_path(str(PROJECTION_SCRIPT), run_name="__main__")

    snapshots = list((tmp_path / "Cross_projection").rglob("launch_config.yaml"))
    assert len(snapshots) == 1
    assert snapshots[0].read_bytes() == RAW_CONFIG


def test_direct_cli_namespace_does_not_create_a_snapshot(tmp_path):
    args = _single_args(run_tag="direct", launch_bytes=None)

    with (
        mock.patch.object(csp, "_load_config", side_effect=RuntimeError("stop after output setup")),
        pytest.raises(RuntimeError, match="stop after output setup"),
    ):
        csp.cross_space_projection(args, out_root=tmp_path)

    assert not list(tmp_path.rglob("launch_config.yaml"))


def test_model_combo_subtrial_gets_its_own_exact_snapshot(tmp_path):
    args = _single_args(run_tag="combo/subtrial_0_0")

    with (
        mock.patch.object(csp, "_load_config", side_effect=RuntimeError("stop after output setup")),
        pytest.raises(RuntimeError, match="stop after output setup"),
    ):
        csp.cross_space_projection(args, out_root=tmp_path)

    snapshots = list(tmp_path.rglob("launch_config.yaml"))
    assert len(snapshots) == 1
    assert "subtrial_0_0" in snapshots[0].parent.name
    assert snapshots[0].read_bytes() == RAW_CONFIG


def test_model_combo_aggregate_gets_an_exact_snapshot_before_reading_subtrials(tmp_path):
    out_dir = tmp_path / "Cross_projection" / "combo" / "aggregated_123"
    missing_result = tmp_path / "missing-results.pkl"
    records = [
        {
            "new_idx": 0,
            "old_idx": 0,
            "new_model_pth": "new.pt",
            "old_model_pth": "old.pt",
            "pkl_path": str(missing_result),
        }
    ]

    with pytest.raises(FileNotFoundError):
        csp._aggregate_model_combo_pkls(records, str(out_dir), _single_args(run_tag="combo"))

    assert (out_dir / "launch_config.yaml").read_bytes() == RAW_CONFIG


def test_grid_root_gets_snapshot_and_records_it_in_best_config(tmp_path):
    study = _CompletedStudy()
    model_config = {
        "model_advanced_params": {"features_folder_saving_path": "unused-features"}
    }

    with (
        mock.patch.object(csp, "_load_config", return_value=model_config),
        mock.patch.object(csp, "_build_model", return_value=object()),
        mock.patch.object(csp, "_precompute_embeddings", return_value=({}, {})),
        mock.patch.object(csp, "_get_sampler", return_value=object()),
        mock.patch.object(csp.optuna, "create_study", return_value=study),
    ):
        out_dir = Path(csp.run_optuna(_grid_args(), out_root=tmp_path))

    assert (out_dir / "launch_config.yaml").read_bytes() == RAW_CONFIG
    assert "config_snapshot: launch_config.yaml\n" in (
        out_dir / "best_config.txt"
    ).read_text(encoding="utf-8")
