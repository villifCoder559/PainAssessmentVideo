from __future__ import annotations

import json
import pickle
import os
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import plot_anchor_sweep_mae as anchor_analysis
from plot_anchor_sweep_mae import (
    DIRECTION_ALIASES,
    PlotOptions,
    discover_groups,
    extract_group,
    render_plots,
    select_best_anchors,
    summarize_subtrials,
    validate_group,
)


METHODS = ("linear", "linear_close", "mlp", "autoencoder", "procrustes")
REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_pickle(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        pickle.dump(value, stream)


def _make_aggregate(k_dir: Path, uid: int, method: str, anchors: int) -> Path:
    path = k_dir / f"aggregated_{uid}" / f"results_{uid}.pkl"
    _write_pickle(
        path,
        {
            "config_cross_space_projection": {
                "interpolation_similarity": method,
                "num_anchors": anchors,
            },
            "n_subtrials": 0,
            "subtrial_pkls": [],
            "subtrials": [],
        },
    )
    return path


def _refinement(after_offset: float = 0.0) -> dict[str, object]:
    return {
        "mae_micro_old_oncsv_before": 4.0,
        "mae_macro_old_oncsv_before": 5.0,
        "mae_micro_old_oncsv_after": 3.0 + after_offset,
        "mae_macro_old_oncsv_after": 3.5 + after_offset,
        # These intentionally disagree with the test arrays and must be ignored.
        "mae_micro_new_test_before": 999.0,
        "mae_macro_new_test_before": 999.0,
        "mae_micro_new_test_after": 999.0,
        "mae_macro_new_test_after": 999.0,
        "new_test_eval": {
            "labels": np.array([0.0, 0.0, 0.0, 1.0]),
            "preds_before": np.array([1.0, 2.0, 3.0, 9.0]),
            "preds_after": np.array([0.0, 2.0, 4.0, 3.0 + after_offset]),
        },
    }


def _make_complete_group(
    root: Path,
    *,
    alias: str | None = None,
    method: str = "linear",
    anchors: int = 5,
    actual_anchors: int = 6,
    pairs: list[tuple[int, int]] | None = None,
) -> tuple[Path, list[Path]]:
    alias = alias or next(iter(DIRECTION_ALIASES))
    k_dir = root / alias / f"refinement3_{method}" / f"K{anchors}"
    aggregate_path = k_dir / "aggregated_100" / "results_100.pkl"
    pairs = pairs if pairs is not None else [(new, old) for new in range(5) for old in range(5)]
    subtrial_paths: list[Path] = []
    manifest_paths: list[str] = []
    manifest_rows: list[dict[str, int]] = []
    for index, (new_idx, old_idx) in enumerate(pairs):
        subtrial_dir = k_dir / f"subtrial_{new_idx}_{old_idx}_{index}"
        anchors_path = subtrial_dir / "anchors.csv"
        anchors_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"anchor": range(actual_anchors)}).to_csv(anchors_path, index=False)
        subtrial_path = subtrial_dir / f"results_{index}.pkl"
        _write_pickle(
            subtrial_path,
            {
                "config_cross_space_projection": {
                    "interpolation_similarity": method,
                    "num_anchors": anchors,
                    "anchors_csv_path": str(anchors_path),
                },
                "anchors_csv_path": str(anchors_path),
                "old_model_tensors": {
                    "labels": np.array([0.0, 0.0, 1.0, 1.0, 1.0]),
                    "predictions": np.array([0.0, 2.0, 1.0, 3.0, 4.0]),
                },
                "refinements": {
                    "linear_only": _refinement(0.0),
                    "projector_linear": _refinement(1.0),
                },
            },
        )
        subtrial_paths.append(subtrial_path)
        manifest_paths.append(str(Path("..") / subtrial_dir.name / subtrial_path.name))
        manifest_rows.append({"new_idx": new_idx, "old_idx": old_idx})

    _write_pickle(
        aggregate_path,
        {
            "config_cross_space_projection": {
                "interpolation_similarity": method,
                "num_anchors": anchors,
            },
            "n_subtrials": len(pairs),
            "subtrial_pkls": manifest_paths,
            "subtrials": manifest_rows,
        },
    )
    return aggregate_path, subtrial_paths


def test_discovers_all_direction_aliases_and_methods_with_numeric_k_order(tmp_path):
    root = tmp_path / "anchor_sweep"
    for alias in DIRECTION_ALIASES:
        for method in METHODS:
            for anchors in (100, 5, 20):
                k_dir = root / alias / f"folder-name-is-not-authoritative-{method}" / f"K{anchors}"
                _make_aggregate(k_dir, 1000 + anchors, method, anchors)

    groups = discover_groups(root, aggregate_policy="error")

    assert len(groups) == 6 * 5 * 3
    assert {group.method for group in groups} == set(METHODS)
    assert {group.direction_alias for group in groups} == set(DIRECTION_ALIASES)
    assert [
        group.configured_anchors
        for group in groups
        if group.direction_alias == next(iter(DIRECTION_ALIASES)) and group.method == "linear"
    ] == [5, 20, 100]


def test_duplicate_aggregates_are_retained_as_validation_error_candidates(tmp_path):
    k_dir = tmp_path / "anchor_sweep" / next(iter(DIRECTION_ALIASES)) / "linear" / "K5"
    first = _make_aggregate(k_dir, 10, "linear", 5)
    second = _make_aggregate(k_dir, 20, "linear", 5)

    [group] = discover_groups(tmp_path / "anchor_sweep", aggregate_policy="error")

    assert group.aggregate_path is None
    assert group.aggregate_candidates == (first.resolve(), second.resolve())


def test_latest_policy_selects_highest_numeric_aggregate_uid(tmp_path):
    k_dir = tmp_path / "anchor_sweep" / next(iter(DIRECTION_ALIASES)) / "linear" / "K5"
    _make_aggregate(k_dir, 9, "linear", 5)
    expected = _make_aggregate(k_dir, 101, "linear", 5)
    _make_aggregate(k_dir, 20, "linear", 5)

    [group] = discover_groups(tmp_path / "anchor_sweep", aggregate_policy="latest")

    assert group.aggregate_path == expected.resolve()
    assert group.aggregate_uid == 101


def test_discovers_legacy_aggregate_when_result_uid_differs(tmp_path):
    root = tmp_path / "anchor_sweep"
    k_dir = root / next(iter(DIRECTION_ALIASES)) / "linear" / "K5"
    canonical = _make_aggregate(k_dir, 10, "linear", 5)
    legacy = canonical.with_name("results_11.pkl")
    canonical.rename(legacy)

    [group] = discover_groups(root, aggregate_policy="error")

    assert group.aggregate_candidates == (legacy.resolve(),)
    assert group.aggregate_path == legacy.resolve()
    assert group.aggregate_uid == 10


def test_latest_policy_rejects_multiple_results_in_latest_aggregate_dir(tmp_path):
    root = tmp_path / "anchor_sweep"
    k_dir = root / next(iter(DIRECTION_ALIASES)) / "linear" / "K5"
    _make_aggregate(k_dir, 9, "linear", 5)
    first = _make_aggregate(k_dir, 10, "linear", 5)
    second = first.with_name("results_11.pkl")
    with first.open("rb") as stream:
        _write_pickle(second, pickle.load(stream))

    [group] = discover_groups(root, aggregate_policy="latest")
    validation = validate_group(group)

    assert group.aggregate_candidates == (first.resolve(), second.resolve())
    assert group.aggregate_path is None
    assert any("Duplicate aggregate candidates" in error for error in validation.errors)


def test_latest_policy_does_not_fall_back_from_empty_newest_aggregate_dir(tmp_path):
    root = tmp_path / "anchor_sweep"
    k_dir = root / next(iter(DIRECTION_ALIASES)) / "linear" / "K5"
    _make_aggregate(k_dir, 9, "linear", 5)
    (k_dir / "aggregated_10").mkdir()

    [group] = discover_groups(root, aggregate_policy="latest")
    validation = validate_group(group)

    assert group.aggregate_candidates == ()
    assert group.aggregate_path is None
    assert any("Missing aggregate candidate" in error for error in validation.errors)


@pytest.mark.parametrize("filename", ("results_backup.pkl", "results_12.pkl"))
def test_rejects_malformed_or_far_mismatched_aggregate_result_name(tmp_path, filename):
    root = tmp_path / "anchor_sweep"
    k_dir = root / next(iter(DIRECTION_ALIASES)) / "linear" / "K5"
    invalid = k_dir / "aggregated_10" / filename
    _write_pickle(invalid, {})

    [group] = discover_groups(root, aggregate_policy="error")
    validation = validate_group(group)

    assert group.aggregate_candidates == ()
    assert group.aggregate_path is None
    assert any("Missing aggregate candidate" in error for error in validation.errors)


def test_rejects_multiple_results_when_one_name_is_incompatible(tmp_path):
    root = tmp_path / "anchor_sweep"
    k_dir = root / next(iter(DIRECTION_ALIASES)) / "linear" / "K5"
    canonical = _make_aggregate(k_dir, 10, "linear", 5)
    incompatible = canonical.with_name("results_backup.pkl")
    with canonical.open("rb") as stream:
        _write_pickle(incompatible, pickle.load(stream))

    [group] = discover_groups(root, aggregate_policy="error")
    validation = validate_group(group)

    assert group.aggregate_candidates == (canonical.resolve(), incompatible.resolve())
    assert group.aggregate_path is None
    assert any("Duplicate aggregate candidates" in error for error in validation.errors)


def test_unknown_direction_alias_has_actionable_error(tmp_path):
    _make_aggregate(tmp_path / "unknown-transfer" / "linear" / "K5", 1, "linear", 5)

    with pytest.raises(ValueError, match="Unknown direction alias.*unknown-transfer"):
        discover_groups(tmp_path, aggregate_policy="error")


def test_validates_exact_cartesian_manifest_and_allow_incomplete(tmp_path):
    root = tmp_path / "anchor_sweep"
    _make_complete_group(root)
    [complete] = discover_groups(root, "error")

    strict = validate_group(complete, allow_incomplete=False)

    assert strict.valid
    assert strict.valid_pairs == tuple((new, old) for new in range(5) for old in range(5))

    partial_root = tmp_path / "partial"
    partial_pairs = [(new, old) for new in range(5) for old in range(5)][:-1]
    _make_complete_group(partial_root, pairs=partial_pairs)
    [partial] = discover_groups(partial_root, "error")

    strict_partial = validate_group(partial, allow_incomplete=False)
    allowed_partial = validate_group(partial, allow_incomplete=True)

    assert not strict_partial.valid
    assert any("25 unique fold pairs" in error for error in strict_partial.errors)
    assert allowed_partial.valid
    assert len(allowed_partial.valid_pairs) == 24
    assert any("incomplete" in warning.lower() for warning in allowed_partial.warnings)


def test_duplicate_pairs_and_missing_referenced_pkls_fail_validation(tmp_path):
    root = tmp_path / "anchor_sweep"
    pairs = [(new, old) for new in range(5) for old in range(5)]
    pairs[-1] = pairs[0]
    _, subtrials = _make_complete_group(root, pairs=pairs)
    subtrials[1].unlink()
    [group] = discover_groups(root, "error")

    result = validate_group(group, allow_incomplete=False)

    assert not result.valid
    assert any("duplicate fold pair" in error.lower() for error in result.errors)
    assert any("missing referenced pkl" in error.lower() for error in result.errors)


def test_extracts_both_modes_and_recomputes_exact_micro_macro_mae(tmp_path):
    root = tmp_path / "anchor_sweep"
    _make_complete_group(root)
    [group] = discover_groups(root, "error")

    rows = extract_group(group, allow_incomplete=False)

    assert len(rows) == 50
    linear_only = rows[0]
    assert linear_only.refinement_mode == "linear_only"
    assert linear_only.source_old_micro == pytest.approx(1.4)
    assert linear_only.source_old_macro == pytest.approx((1.0 + 5.0 / 3.0) / 2.0)
    assert linear_only.source_before_micro == 4.0
    assert linear_only.source_after_micro == 3.0
    assert linear_only.target_before_micro == pytest.approx(3.5)
    assert linear_only.target_before_macro == pytest.approx(5.0)
    assert linear_only.target_after_micro == pytest.approx(2.0)
    assert linear_only.target_after_macro == pytest.approx(2.0)
    assert linear_only.source_improvement_micro == pytest.approx(1.0)
    assert linear_only.target_preservation_change_micro == pytest.approx(1.5)
    assert linear_only.actual_anchors == 6
    assert linear_only.anchor_count_mismatch
    assert {row.refinement_mode for row in rows} == {"linear_only", "projector_linear"}


def test_summary_uses_arithmetic_mean_sample_sd_and_actual_n(tmp_path):
    root = tmp_path / "anchor_sweep"
    _make_complete_group(root)
    [group] = discover_groups(root, "error")
    rows = pd.DataFrame(asdict(row) for row in extract_group(group, False))
    rows = rows.loc[rows["refinement_mode"] == "linear_only"].iloc[:3].copy()
    rows["source_after_micro"] = [1.0, 2.0, 3.0]

    summary = summarize_subtrials(rows)
    selected = summary[
        (summary["metric"] == "micro") & (summary["stage"] == "source_after")
    ].iloc[0]

    assert selected["mean"] == pytest.approx(2.0)
    assert selected["sd"] == pytest.approx(1.0)
    assert selected["n"] == 3


def test_best_anchor_selection_uses_final_source_mae_and_smaller_k_tie_break():
    common = {
        "direction_alias": "bioVmae_to_mintDfer",
        "direction": "biovid-to-mintpain",
        "source_dataset": "BioVid",
        "target_dataset": "MIntPain",
        "method": "linear",
        "refinement_mode": "linear_only",
        "metric": "micro",
        "stage": "source_after",
        "sd": 0.1,
        "n": 25,
    }
    summary = pd.DataFrame(
        [
            {**common, "configured_anchors": 100, "mean": 1.0},
            {**common, "configured_anchors": 5, "mean": 1.0 + 5e-13},
            {**common, "configured_anchors": 20, "mean": 1.2},
            {**common, "stage": "target_after", "configured_anchors": 1, "mean": 0.1},
        ]
    )

    best = select_best_anchors(summary)

    assert len(best) == 1
    assert best.iloc[0]["configured_anchors"] == 5
    assert best.iloc[0]["mean"] == pytest.approx(1.0 + 5e-13)


def test_render_plots_expands_line_scales_and_formats(tmp_path):
    root = tmp_path / "anchor_sweep"
    _make_complete_group(root)
    [group] = discover_groups(root, "error")
    subtrials = pd.DataFrame(asdict(row) for row in extract_group(group, False))
    second_k = subtrials.copy()
    second_k["configured_anchors"] = 20
    second_k["source_after_micro"] += 0.25
    subtrials = pd.concat([subtrials, second_k], ignore_index=True)
    summary = summarize_subtrials(subtrials)
    options = PlotOptions(
        output_dir=tmp_path / "output",
        metrics=("micro",),
        anchor_scales=("log", "categorical"),
        plots=("stages",),
        formats=("png", "pdf"),
        dpi=72,
    )

    paths = render_plots(subtrials, summary, options)

    assert len(paths) == 4
    assert {path.suffix for path in paths} == {".png", ".pdf"}
    assert any("__log." in path.name for path in paths)
    assert any("__categorical." in path.name for path in paths)
    assert all(path.is_file() and path.stat().st_size > 0 for path in paths)


def test_render_plots_excludes_linear_close_from_every_plot_family(tmp_path):
    root = tmp_path / "anchor_sweep"
    _make_complete_group(root, method="linear")
    _make_complete_group(root, method="linear_close")
    groups = discover_groups(root, "error")
    subtrials = pd.concat(
        [pd.DataFrame(asdict(row) for row in extract_group(group, False)) for group in groups],
        ignore_index=True,
    )
    summary = summarize_subtrials(subtrials)
    options = PlotOptions(
        output_dir=tmp_path / "output",
        metrics=("micro",),
        anchor_scales=("categorical",),
        plots=("stages", "methods", "improvement", "heatmap", "distribution"),
        formats=("png",),
        dpi=72,
        exclude_linear_close=True,
        show_progress=False,
    )

    paths = render_plots(subtrials, summary, options)

    assert len(paths) == 5
    assert {path.name.split("__", 1)[0] for path in paths} == {
        "stages",
        "methods",
        "improvement",
        "heatmap",
        "distribution",
    }
    assert all("linear_close" not in path.name for path in paths)


def test_exclusion_reaches_shared_figure_contents_and_progress(tmp_path, monkeypatch):
    root = tmp_path / "anchor_sweep"
    _make_complete_group(root, method="linear")
    _make_complete_group(root, method="linear_close")
    groups = discover_groups(root, "error")
    subtrials = pd.concat(
        [pd.DataFrame(asdict(row) for row in extract_group(group, False)) for group in groups],
        ignore_index=True,
    )
    summary = summarize_subtrials(subtrials)
    default_options = PlotOptions(output_dir=tmp_path / "default")
    assert default_options.exclude_linear_close is False
    default_methods = anchor_analysis._methods_figure(
        summary, "biovid-to-mintpain", "micro", "categorical"
    )
    default_heatmap = anchor_analysis._heatmap_figure(
        summary, "biovid-to-mintpain", "micro"
    )
    try:
        assert "Linear close" in {
            line.get_label() for axis in default_methods.axes for line in axis.lines
        }
        assert "Linear close" in {
            label.get_text()
            for axis in default_heatmap.axes[:3]
            for label in axis.get_yticklabels()
        }
    finally:
        import matplotlib.pyplot as plt

        plt.close(default_methods)
        plt.close(default_heatmap)

    progress_state: dict[str, int] = {}

    class RecordingProgress:
        def update(self, amount: int) -> None:
            progress_state["updates"] = progress_state.get("updates", 0) + amount

        def close(self) -> None:
            progress_state["closed"] = 1

    def record_progress(*_args, total=None, **_kwargs):
        progress_state["total"] = int(total)
        return RecordingProgress()

    inspected_families: set[str] = set()

    def inspect_figure(figure, base, _options):
        family = base.name.split("__", 1)[0]
        inspected_families.add(family)
        if family in {"methods", "improvement"}:
            labels = {line.get_label() for axis in figure.axes for line in axis.lines}
            assert "Linear close" not in labels
        if family == "heatmap":
            labels = {
                label.get_text()
                for axis in figure.axes[:3]
                for label in axis.get_yticklabels()
            }
            assert "Linear close" not in labels
        if family in {"stages", "distribution"}:
            assert "linear_close" not in base.name
        return [base.with_suffix(".png")]

    monkeypatch.setattr(anchor_analysis, "_progress", record_progress)
    monkeypatch.setattr(anchor_analysis, "_save_figure", inspect_figure)
    options = PlotOptions(
        output_dir=tmp_path / "output",
        metrics=("micro",),
        anchor_scales=("categorical",),
        plots=("stages", "methods", "improvement", "heatmap", "distribution"),
        formats=("png",),
        dpi=72,
        exclude_linear_close=True,
        show_progress=False,
    )

    paths = render_plots(subtrials, summary, options)

    assert len(paths) == 5
    assert inspected_families == {
        "stages",
        "methods",
        "improvement",
        "heatmap",
        "distribution",
    }
    assert progress_state == {"total": 5, "updates": 5, "closed": 1}


def test_methods_figure_compares_source_and_target_mae_for_all_plot_stages():
    curves = {
        ("linear_only", "source_before"): (1.0, 2.0),
        ("projector_linear", "source_before"): (1.0, 2.0),
        ("linear_only", "target_before"): (3.0, 4.0),
        ("projector_linear", "target_before"): (3.0, 4.0),
        ("linear_only", "source_after"): (5.0, 6.0),
        ("projector_linear", "source_after"): (7.0, 8.0),
        ("linear_only", "target_after"): (9.0, 10.0),
        ("projector_linear", "target_after"): (11.0, 12.0),
    }
    records = [
        {
            "direction": "biovid-to-mintpain",
            "source_dataset": "BioVid",
            "target_dataset": "MIntPain",
            "metric": "micro",
            "stage": stage,
            "refinement_mode": mode,
            "method": "linear",
            "configured_anchors": anchors,
            "mean": mean,
            "sd": 0.1,
            "n": 25,
        }
        for (mode, stage), values in curves.items()
        for anchors, mean in zip((5, 20), values)
    ]
    summary = pd.DataFrame.from_records(records)

    figure = anchor_analysis._methods_figure(
        summary, "biovid-to-mintpain", "micro", "categorical"
    )
    try:
        assert len(figure.axes) == 6
        expected = (
            (1.0, 2.0),
            (5.0, 6.0),
            (7.0, 8.0),
            (3.0, 4.0),
            (9.0, 10.0),
            (11.0, 12.0),
        )
        for axis, values in zip(figure.axes, expected):
            [line] = axis.lines
            assert line.get_label() == "Linear"
            assert line.get_ydata().tolist() == list(values)
        assert [axis.get_title() for axis in figure.axes[:3]] == [
            "Projector only",
            "Linear only",
            "Projector + linear",
        ]
        assert figure.axes[0].get_ylabel() == "BioVid micro-MAE"
        assert figure.axes[3].get_ylabel() == "MIntPain micro-MAE"
        assert figure._suptitle.get_text() == (
            "biovid-to-mintpain: MAE by method and refinement stage"
        )
        for axis in figure.axes:
            assert axis.get_xlabel() == "number of anchors"
            numeric_labels = [
                label for label in axis.get_yticklabels() if label.get_text()
            ]
            assert numeric_labels
            assert all(label.get_visible() for label in numeric_labels)
    finally:
        import matplotlib.pyplot as plt

        plt.close(figure)


def test_projector_only_summary_rows_collapse_matching_before_values():
    summary = pd.DataFrame.from_records(
        [
            {
                "direction": "biovid-to-mintpain",
                "source_dataset": "BioVid",
                "target_dataset": "MIntPain",
                "metric": "micro",
                "stage": stage,
                "refinement_mode": mode,
                "method": "linear",
                "configured_anchors": anchors,
                "mean": mean,
                "sd": 0.1,
                "n": 25,
            }
            for mode in ("linear_only", "projector_linear")
            for stage, means in (
                ("source_before", (1.4, 1.2)),
                ("target_before", (0.8, 0.7)),
            )
            for anchors, mean in zip((5, 20), means)
        ]
    )

    source = anchor_analysis._summary_rows_for_plot_stage(
        summary, "projector_only", "source_after"
    )
    target = anchor_analysis._summary_rows_for_plot_stage(
        summary, "projector_only", "target_after"
    )

    assert source["mean"].tolist() == [1.4, 1.2]
    assert target["mean"].tolist() == [0.8, 0.7]
    assert source["refinement_mode"].tolist() == ["projector_only"] * 2
    assert target["refinement_mode"].tolist() == ["projector_only"] * 2


def test_projector_only_summary_rows_reject_inconsistent_before_values():
    summary = pd.DataFrame.from_records(
        [
            {
                "direction": "biovid-to-mintpain",
                "source_dataset": "BioVid",
                "target_dataset": "MIntPain",
                "metric": "micro",
                "stage": "source_before",
                "refinement_mode": mode,
                "method": "linear",
                "configured_anchors": 5,
                "mean": mean,
                "sd": 0.1,
                "n": 25,
            }
            for mode, mean in (("linear_only", 1.4), ("projector_linear", 1.5))
        ]
    )

    with pytest.raises(ValueError, match="Inconsistent projector-only.*mean"):
        anchor_analysis._summary_rows_for_plot_stage(
            summary, "projector_only", "source_after"
        )


def test_render_rejects_any_inconsistent_projector_only_before_value(tmp_path):
    root = tmp_path / "anchor_sweep"
    _make_complete_group(root)
    [group] = discover_groups(root, "error")
    subtrials = pd.DataFrame(asdict(row) for row in extract_group(group, False))
    mismatch = (
        (subtrials["refinement_mode"] == "projector_linear")
        & (subtrials["new_idx"] == 0)
        & (subtrials["old_idx"] == 0)
    )
    subtrials.loc[mismatch, "target_before_micro"] += 0.5
    summary = summarize_subtrials(subtrials)
    options = PlotOptions(
        output_dir=tmp_path / "output",
        metrics=("micro",),
        plots=("heatmap",),
        formats=("png",),
        dpi=72,
        show_progress=False,
    )

    with pytest.raises(ValueError, match="target_before_micro"):
        render_plots(subtrials, summary, options)


def test_compatible_figures_include_projector_only_stage(tmp_path):
    root = tmp_path / "anchor_sweep"
    _make_complete_group(root)
    [group] = discover_groups(root, "error")
    subtrials = pd.DataFrame(asdict(row) for row in extract_group(group, False))
    summary = summarize_subtrials(subtrials)

    figures = (
        anchor_analysis._stage_figure(
            summary, "biovid-to-mintpain", "linear", "micro", "categorical"
        ),
        anchor_analysis._heatmap_figure(summary, "biovid-to-mintpain", "micro"),
        anchor_analysis._distribution_figure(
            subtrials, "biovid-to-mintpain", "linear", "micro"
        ),
    )
    try:
        stage_figure, heatmap_figure, distribution_figure = figures
        assert len(stage_figure.axes) == 6
        assert [axis.get_title() for axis in stage_figure.axes[:3]] == [
            "Projector only",
            "Linear only",
            "Projector + linear",
        ]
        assert len(heatmap_figure.axes) == 4  # three panels plus colorbar
        assert [axis.get_title() for axis in heatmap_figure.axes[:3]] == [
            "Projector only",
            "Linear only",
            "Projector + linear",
        ]
        assert len(distribution_figure.axes) == 3
        assert [axis.get_title() for axis in distribution_figure.axes] == [
            "Projector only",
            "Linear only",
            "Projector + linear",
        ]
        assert distribution_figure.axes[0].get_ylabel() == "Source-test micro-MAE"
        for axis in stage_figure.axes:
            assert axis.get_xlabel() == "number of anchors"
        for axis in heatmap_figure.axes[:3]:
            assert axis.get_xlabel() == "number of anchors"
        for axis in distribution_figure.axes:
            assert axis.get_xlabel() == "number of anchors"
            numeric_labels = [
                label for label in axis.get_yticklabels() if label.get_text()
            ]
            assert numeric_labels
            assert all(label.get_visible() for label in numeric_labels)
    finally:
        import matplotlib.pyplot as plt

        for figure in figures:
            plt.close(figure)


def test_categorical_only_plots_are_not_duplicated_for_both_scales(tmp_path):
    root = tmp_path / "anchor_sweep"
    _make_complete_group(root)
    [group] = discover_groups(root, "error")
    subtrials = pd.DataFrame(asdict(row) for row in extract_group(group, False))
    summary = summarize_subtrials(subtrials)
    options = PlotOptions(
        output_dir=tmp_path / "output",
        metrics=("micro",),
        anchor_scales=("log", "categorical"),
        plots=("heatmap", "distribution"),
        formats=("png",),
        dpi=72,
    )

    paths = render_plots(subtrials, summary, options)

    assert len(paths) == 2
    assert {path.name.split("__", 1)[0] for path in paths} == {"heatmap", "distribution"}


def test_standard_deviation_ribbons_are_opt_in(tmp_path):
    root = tmp_path / "anchor_sweep"
    _make_complete_group(root)
    [group] = discover_groups(root, "error")
    subtrials = pd.DataFrame(asdict(row) for row in extract_group(group, False))
    summary = summarize_subtrials(subtrials)

    hidden = anchor_analysis._methods_figure(
        summary, "biovid-to-mintpain", "micro", "categorical"
    )
    shown = anchor_analysis._methods_figure(
        summary, "biovid-to-mintpain", "micro", "categorical", show_std=True
    )
    try:
        assert all(not axis.collections for axis in hidden.axes)
        assert all(axis.collections for axis in shown.axes)
    finally:
        import matplotlib.pyplot as plt

        plt.close(hidden)
        plt.close(shown)


def test_improvement_standard_deviation_ribbons_are_not_clipped_at_zero():
    summary = pd.DataFrame.from_records(
        [
            {
                "direction": "biovid-to-mintpain",
                "metric": "micro",
                "stage": stage,
                "refinement_mode": mode,
                "method": "linear",
                "configured_anchors": anchors,
                "mean": -2.0,
                "sd": 0.5,
            }
            for mode in ("linear_only", "projector_linear")
            for stage in ("source_improvement", "target_preservation_change")
            for anchors in (5, 20)
        ]
    )

    figure = anchor_analysis._improvement_figure(
        summary, "biovid-to-mintpain", "micro", "categorical", show_std=True
    )
    try:
        ribbon_y = figure.axes[0].collections[0].get_paths()[0].vertices[:, 1]
        assert ribbon_y.min() == pytest.approx(-2.5)
        assert ribbon_y.max() == pytest.approx(-1.5)
    finally:
        import matplotlib.pyplot as plt

        plt.close(figure)


def _run_module(*arguments: object) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["MPLCONFIGDIR"] = "/tmp/mplconfig"
    return subprocess.run(
        [sys.executable, "-m", "plot_anchor_sweep_mae", *map(str, arguments)],
        cwd=REPO_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )


def test_module_help_lists_core_cli_modes():
    result = _run_module("--help")
    help_text = " ".join(result.stdout.split())

    assert result.returncode == 0, result.stderr
    assert "--aggregate-policy" in help_text
    assert "exactly one aggregate" in help_text
    assert "highest numeric aggregate UID" in help_text
    assert "--allow-incomplete" in help_text
    assert "--extract-only" in help_text
    assert "--plot-only" in help_text
    assert "--show-std" in help_text
    assert "--exclude-linear-close" in help_text


def test_exclude_linear_close_preserves_tables_and_filters_figures(tmp_path):
    root = tmp_path / "anchor_sweep"
    _make_complete_group(root, method="linear")
    _make_complete_group(root, method="linear_close")
    output = tmp_path / "results"

    result = _run_module(
        "--input-root",
        root,
        "--output-dir",
        output,
        "--exclude-linear-close",
        "--metric",
        "micro",
        "--anchor-scale",
        "categorical",
        "--format",
        "png",
        "--dpi",
        72,
        "--workers",
        1,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    summary = pd.read_csv(output / "data" / "anchor_sweep_summary.csv")
    subtrials = pd.read_csv(output / "data" / "anchor_sweep_subtrials.csv")
    best = pd.read_csv(output / "data" / "anchor_sweep_best_anchors.csv")
    validation = pd.read_csv(output / "data" / "validation_report.csv")
    assert set(summary["method"]) == {"linear", "linear_close"}
    assert set(subtrials["method"]) == {"linear", "linear_close"}
    assert set(best["method"]) == {"linear", "linear_close"}
    assert set(validation.loc[validation["method"].notna(), "method"]) == {
        "linear",
        "linear_close",
    }
    cache_names = {path.name for path in (output / "cache").iterdir()}
    assert {
        "biovid-to-mintpain__linear__K5.csv",
        "biovid-to-mintpain__linear__K5.json",
        "biovid-to-mintpain__linear_close__K5.csv",
        "biovid-to-mintpain__linear_close__K5.json",
    } <= cache_names
    figures = sorted((output / "figures" / "micro").glob("*.png"))
    assert len(figures) == 5
    assert all("linear_close" not in path.name for path in figures)
    with (output / "data" / "run_metadata.json").open(encoding="utf-8") as stream:
        metadata = json.load(stream)
    assert metadata["arguments"]["exclude_linear_close"] is True
    assert metadata["subtrial_rows"] == 100
    assert len(metadata["figures"]) == 5
    assert all("linear_close" not in Path(path).name for path in metadata["figures"])


def test_exclude_linear_close_conflicts_with_explicit_method_selection():
    result = _run_module(
        "--method",
        "linear_close",
        "mlp",
        "--exclude-linear-close",
    )

    assert result.returncode != 0
    assert "--exclude-linear-close cannot be combined" in result.stderr


def test_exclude_linear_close_is_a_noop_for_extract_only(tmp_path):
    root = tmp_path / "anchor_sweep"
    _make_complete_group(root, method="linear_close")
    output = tmp_path / "results"

    result = _run_module(
        "--input-root",
        root,
        "--output-dir",
        output,
        "--method",
        "linear_close",
        "--exclude-linear-close",
        "--extract-only",
        "--workers",
        1,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    summary = pd.read_csv(output / "data" / "anchor_sweep_summary.csv")
    assert set(summary["method"]) == {"linear_close"}


def test_inconsistent_projector_only_values_fail_before_cache_or_tables(tmp_path):
    root = tmp_path / "anchor_sweep"
    _, subtrial_paths = _make_complete_group(root)
    with subtrial_paths[0].open("rb") as stream:
        payload = pickle.load(stream)
    payload["refinements"]["projector_linear"]["mae_micro_old_oncsv_before"] += 0.5
    _write_pickle(subtrial_paths[0], payload)
    output = tmp_path / "results"

    result = _run_module(
        "--input-root",
        root,
        "--output-dir",
        output,
        "--extract-only",
        "--workers",
        1,
    )

    assert result.returncode != 0
    assert "Traceback" not in result.stderr
    report = pd.read_csv(output / "data" / "validation_report.csv")
    assert report["message"].str.contains("Inconsistent projector-only").any()
    assert not list((output / "cache").glob("*.csv"))
    assert not (output / "data" / "anchor_sweep_subtrials.csv").exists()


def test_plot_only_reports_inconsistent_cached_before_values_cleanly(tmp_path):
    root = tmp_path / "anchor_sweep"
    _make_complete_group(root)
    output = tmp_path / "results"
    extraction = _run_module(
        "--input-root",
        root,
        "--output-dir",
        output,
        "--extract-only",
        "--workers",
        1,
    )
    assert extraction.returncode == 0, extraction.stdout + extraction.stderr
    cache_csv = output / "cache" / "biovid-to-mintpain__linear__K5.csv"
    cached = pd.read_csv(cache_csv)
    mismatch = (
        (cached["refinement_mode"] == "projector_linear")
        & (cached["new_idx"] == 0)
        & (cached["old_idx"] == 0)
    )
    cached.loc[mismatch, "target_before_micro"] += 0.5
    cached.to_csv(cache_csv, index=False)
    for name in (
        "anchor_sweep_subtrials.csv",
        "anchor_sweep_summary.csv",
        "anchor_sweep_best_anchors.csv",
    ):
        (output / "data" / name).unlink()

    result = _run_module(
        "--output-dir",
        output,
        "--plot-only",
        "--plots",
        "heatmap",
        "--metric",
        "micro",
        "--format",
        "png",
    )

    assert result.returncode != 0
    assert "Traceback" not in result.stderr
    report = pd.read_csv(output / "data" / "validation_report.csv")
    assert report["message"].str.contains("target_before_micro").any()
    assert not (output / "data" / "anchor_sweep_subtrials.csv").exists()


def test_cli_writes_tables_reuses_cache_and_invalidates_changed_subtrial(tmp_path):
    root = tmp_path / "anchor_sweep"
    _, subtrial_paths = _make_complete_group(root)
    output = tmp_path / "results"
    arguments = (
        "--input-root",
        root,
        "--output-dir",
        output,
        "--direction",
        "biovid-to-mintpain",
        "--method",
        "linear",
        "--anchors",
        5,
        "--workers",
        1,
        "--extract-only",
    )

    first = _run_module(*arguments)

    assert first.returncode == 0, first.stdout + first.stderr
    data_dir = output / "data"
    expected_tables = {
        "anchor_sweep_subtrials.csv",
        "anchor_sweep_summary.csv",
        "anchor_sweep_best_anchors.csv",
        "validation_report.csv",
        "run_metadata.json",
    }
    assert expected_tables <= {path.name for path in data_dir.iterdir()}
    extracted = pd.read_csv(data_dir / "anchor_sweep_subtrials.csv")
    assert len(extracted) == 50
    cache_csv = output / "cache" / "biovid-to-mintpain__linear__K5.csv"
    cache_json = cache_csv.with_suffix(".json")
    assert cache_csv.is_file() and cache_json.is_file()
    first_mtime = cache_csv.stat().st_mtime_ns

    second = _run_module(*arguments)
    assert second.returncode == 0, second.stdout + second.stderr
    assert cache_csv.stat().st_mtime_ns == first_mtime

    time.sleep(0.01)
    os.utime(subtrial_paths[0], None)
    third = _run_module(*arguments)
    assert third.returncode == 0, third.stdout + third.stderr
    assert cache_csv.stat().st_mtime_ns > first_mtime
    third_mtime = cache_csv.stat().st_mtime_ns

    time.sleep(0.01)
    pd.DataFrame({"anchor": range(3)}).to_csv(
        subtrial_paths[0].parent / "anchors.csv", index=False
    )
    fourth = _run_module(*arguments)
    assert fourth.returncode == 0, fourth.stdout + fourth.stderr
    assert cache_csv.stat().st_mtime_ns > third_mtime
    refreshed = pd.read_csv(output / "data" / "anchor_sweep_subtrials.csv")
    changed_pair = refreshed[(refreshed["new_idx"] == 0) & (refreshed["old_idx"] == 0)]
    assert set(changed_pair["actual_anchors"]) == {3}


def test_plot_only_uses_compatible_cache_without_scanning_input_root(tmp_path):
    root = tmp_path / "anchor_sweep"
    _make_complete_group(root)
    output = tmp_path / "results"
    extraction = _run_module(
        "--input-root",
        root,
        "--output-dir",
        output,
        "--workers",
        1,
        "--extract-only",
    )
    assert extraction.returncode == 0, extraction.stdout + extraction.stderr
    root.rename(tmp_path / "moved-away")

    plotting = _run_module(
        "--input-root",
        root,
        "--output-dir",
        output,
        "--plot-only",
        "--direction",
        "biovid-to-mintpain",
        "--method",
        "linear",
        "--anchors",
        5,
        "--metric",
        "micro",
        "--anchor-scale",
        "categorical",
        "--plots",
        "methods",
        "--format",
        "png",
        "--dpi",
        72,
    )

    assert plotting.returncode == 0, plotting.stdout + plotting.stderr
    figures = list((output / "figures" / "micro").glob("*.png"))
    assert [path.name for path in figures] == [
        "methods__biovid-to-mintpain__categorical.png"
    ]


def test_strict_validation_exits_nonzero_before_plotting(tmp_path):
    root = tmp_path / "anchor_sweep"
    pairs = [(new, old) for new in range(5) for old in range(5)][:-1]
    _make_complete_group(root, pairs=pairs)
    output = tmp_path / "results"

    result = _run_module(
        "--input-root",
        root,
        "--output-dir",
        output,
        "--format",
        "png",
        "--metric",
        "micro",
    )

    assert result.returncode != 0
    report = pd.read_csv(output / "data" / "validation_report.csv")
    assert (report["severity"] == "error").any()
    assert not (output / "figures").exists()


def test_corrupt_aggregate_is_reported_as_validation_error(tmp_path):
    root = tmp_path / "anchor_sweep"
    alias = next(iter(DIRECTION_ALIASES))
    aggregate = root / alias / "refinement3_linear" / "K5" / "aggregated_100" / "results_100.pkl"
    aggregate.parent.mkdir(parents=True)
    aggregate.write_bytes(b"not a pickle")
    output = tmp_path / "results"

    result = _run_module("--input-root", root, "--output-dir", output, "--extract-only")

    assert result.returncode != 0
    assert "Traceback" not in result.stderr
    report = pd.read_csv(output / "data" / "validation_report.csv")
    assert (report["severity"] == "error").any()
    assert report["message"].str.contains("Could not read aggregate PKL").any()


def test_missing_anchors_csv_is_reported_before_extraction(tmp_path):
    root = tmp_path / "anchor_sweep"
    _, subtrials = _make_complete_group(root)
    (subtrials[0].parent / "anchors.csv").unlink()
    output = tmp_path / "results"

    result = _run_module("--input-root", root, "--output-dir", output, "--extract-only")

    assert result.returncode != 0
    assert "Traceback" not in result.stderr
    report = pd.read_csv(output / "data" / "validation_report.csv")
    assert (report["severity"] == "error").any()
    assert report["message"].str.contains("Missing anchors.csv").any()
    assert not list((output / "cache").glob("*.csv"))


def test_allow_incomplete_skips_a_fold_with_invalid_local_inputs(tmp_path):
    root = tmp_path / "anchor_sweep"
    _, subtrials = _make_complete_group(root)
    (subtrials[0].parent / "anchors.csv").unlink()
    [group] = discover_groups(root, "error")

    result = validate_group(group, allow_incomplete=True)
    rows = extract_group(group, allow_incomplete=True)

    assert result.valid
    assert len(result.valid_pairs) == 24
    assert len(rows) == 48
    assert any("Skipping invalid fold pair" in warning for warning in result.warnings)


def test_interactive_analysis_shows_phase_progress_and_eta(tmp_path, monkeypatch, capsys):
    root = tmp_path / "anchor_sweep"
    _make_complete_group(root)
    output = tmp_path / "results"
    monkeypatch.setattr(anchor_analysis, "_progress_enabled", lambda: True)

    result = anchor_analysis.main(
        [
            "--input-root",
            str(root),
            "--output-dir",
            str(output),
            "--workers",
            "1",
            "--metric",
            "micro",
            "--anchor-scale",
            "categorical",
            "--plots",
            "methods",
            "--format",
            "png",
            "--dpi",
            "72",
        ]
    )
    stderr = capsys.readouterr().err

    assert result == 0
    assert "Validating groups" in stderr
    assert "Extracting groups" in stderr
    assert "Rendering figures" in stderr
    assert "ETA" in stderr
