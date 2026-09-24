from pathlib import Path
import csv
import hashlib
import os
import pickle
import shlex
import sys
from types import SimpleNamespace

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import plot_cumulative_predictions as pcp


def test_plot_cumulative_predictions_script_exists():
    assert (REPO_ROOT / "plot_cumulative_predictions.py").is_file()


def test_chunk_frames_uses_sixteen_frame_chunks_and_repeats_final_frame():
    frames = np.arange(33, dtype=np.uint8)[:, None]

    chunks = pcp.chunk_frames(frames)

    assert chunks.shape == (3, 16, 1)
    assert chunks[0, :, 0].tolist() == list(range(16))
    assert chunks[1, :, 0].tolist() == list(range(16, 32))
    assert chunks[2, :, 0].tolist() == [32] * 16


def test_chunk_frames_rejects_empty_video():
    with pytest.raises(ValueError, match="no frames"):
        pcp.chunk_frames(np.empty((0, 2), dtype=np.uint8))


def test_prefixes_and_cumulative_frame_ranges_use_zero_based_indices():
    chunks = np.arange(3 * 16).reshape(3, 16)

    prefixes = pcp.cumulative_prefixes(chunks)

    assert [len(prefix) for prefix in prefixes] == [1, 2, 3]
    assert pcp.cumulative_frame_ranges(33) == [(0, 15), (0, 31), (0, 32)]


@pytest.mark.parametrize(
    ("stage", "mode"),
    [(1, "linear_only"), (2, "projector_linear"),
     (4, "random_projector_linear")],
)
def test_refinement_stage_mapping(stage, mode):
    assert pcp.refinement_mode(stage) == mode


def test_curve_averaging_is_elementwise_and_validates_lengths():
    assert np.allclose(
        pcp.mean_curve([[1.0, 2.0], [3.0, 6.0]]),
        [2.0, 4.0],
    )
    with pytest.raises(ValueError, match="same length"):
        pcp.mean_curve([[1.0], [1.0, 2.0]])


def test_cross_output_directory_contains_model_pairs_and_projector_kind(tmp_path):
    path = pcp.invocation_output_dir(
        tmp_path,
        sample_id="123",
        origin="source",
        stage=2,
        origin_dataset="BIOVID",
        origin_model="VideoMAE",
        source_dataset="BIOVID",
        source_model="VideoMAE",
        target_dataset="UNBC",
        target_model="DFER",
        projector_kind="mlp",
    )

    assert path == (
        tmp_path / "videomae_biovid_to_unbc_dfer" / "123"
        / "mlp" / "videomae_biovid" / "stage_2"
    )


def test_native_output_path_omits_refinement_stage(tmp_path):
    path = pcp.invocation_output_dir(
        tmp_path,
        sample_id="59",
        origin="native",
        stage=None,
        origin_dataset="UNBC",
        origin_model="VideoMAE",
    )

    assert path == tmp_path / "videomae_unbc" / "59" / "native"


def test_cross_target_output_directory_names_the_target_origin(tmp_path):
    path = pcp.invocation_output_dir(
        tmp_path,
        sample_id="59",
        origin="target",
        stage=1,
        origin_dataset="UNBC",
        origin_model="DFER",
        source_dataset="BIOVID",
        source_model="VideoMAE",
        target_dataset="UNBC",
        target_model="DFER",
        projector_kind="linear",
    )

    assert path == (
        tmp_path / "videomae_biovid_to_unbc_dfer" / "59"
        / "linear" / "dfer_unbc" / "stage_1"
    )


def test_cross_output_directory_requires_projector_kind(tmp_path):
    with pytest.raises(ValueError, match="projector kind"):
        pcp.invocation_output_dir(
            tmp_path,
            sample_id="59",
            origin="target",
            stage=1,
            origin_dataset="UNBC",
            origin_model="DFER",
            source_dataset="BIOVID",
            source_model="VideoMAE",
            target_dataset="UNBC",
            target_model="DFER",
        )


def test_invocation_digest_is_stable_and_argument_sensitive():
    argv = ["experiment root", "59", "--origin", "target"]

    assert pcp.invocation_digest(argv) == pcp.invocation_digest(argv)
    assert pcp.invocation_digest(argv) != pcp.invocation_digest([
        *argv, "--refinement-stage", "1",
    ])
    assert len(pcp.invocation_digest(argv)) == 12


@pytest.mark.parametrize(
    ("model_type", "dataset_path", "expected"),
    [
        ("VIDEOMAE_v2_S", "UNBC/video/features/VideoMaev2_S", ("UNBC", "VideoMAE")),
        ("DFER", "partA/video/features/DFER/Biovid", ("BIOVID", "DFER")),
        ("VIDEOMAE_v2_G", "MIntPAIN/features/VideoMAE", ("MINTPAIN", "VideoMAE")),
        ("DFER", "PEMF/video/features/DFER", ("PEMF", "DFER")),
    ],
)
def test_model_dataset_identity_uses_short_model_family(
        model_type, dataset_path, expected):
    config = {"model_advanced_params": {
        "model_type": model_type,
        "features_folder_saving_path": dataset_path,
    }}

    assert pcp.model_dataset_identity(config) == expected


def test_model_dataset_identity_rejects_unknown_dataset_with_supported_names():
    config = {"model_advanced_params": {
        "model_type": "DFER",
        "path_dataset": "custom_dataset/videos",
    }}

    with pytest.raises(ValueError, match=r"Cannot detect dataset.*UNBC.*BIOVID.*PEMF"):
        pcp.model_dataset_identity(config)


def test_inverse_target_normalization_does_not_clip():
    config = {"config": {"target_spec": {
        "target_min": 2.0,
        "target_max": 6.0,
        "normalization": "min_max",
    }}}
    assert np.allclose(pcp.inverse_target([-0.5, 1.5], config), [0.0, 8.0])


@pytest.mark.parametrize("config", [
    {"config": {"target_spec": {"target_max": 6.26}}},
    {"config": {"max_label": 6.26}},
])
def test_dataset_axis_max_prefers_configured_target_range(tmp_path, config):
    csv_path = tmp_path / "train_HEAD" / "k0_cross_val" / "test.csv"

    assert pcp.dataset_axis_max(config, csv_path) == pytest.approx(6.26)


def _write_tsv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["subject_id", "subject_name", "class_id", "class_name",
                        "sample_id", "sample_name"],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)


def _row(sample_id, sample_name="clip", subject="subject", pain=3.0):
    return {
        "subject_id": "1",
        "subject_name": subject,
        "class_id": str(pain),
        "class_name": str(pain),
        "sample_id": str(sample_id),
        "sample_name": sample_name,
    }


def test_dataset_axis_max_falls_back_to_all_outer_fold_test_csvs(tmp_path):
    train_root = tmp_path / "train_HEAD"
    first = train_root / "k0_cross_val" / "test.csv"
    _write_tsv(first, [_row(1, pain=0.0), _row(2, pain=4.5)])
    _write_tsv(
        train_root / "k1_cross_val" / "test.csv",
        [_row(3, pain=6.25)],
    )
    _write_tsv(
        train_root / "k1_cross_val" / "k1_cross_val_sub_0" / "test.csv",
        [_row(4, pain=99.0)],
    )

    assert pcp.dataset_axis_max({"config": {}}, first) == pytest.approx(6.25)


def _native_experiment(tmp_path, fold_samples=((11,), (42,))):
    root = tmp_path / "native"
    videos = tmp_path / "videos"
    results = {}
    epochs = (4, 7)
    subfolds = (1, 2)
    for fold, sample_ids in enumerate(fold_samples):
        fold_dir = root / "train_ATTENTIVE_JEPA" / f"k{fold}_cross_val"
        _write_tsv(fold_dir / "test.csv", [_row(sid) for sid in sample_ids])
        checkpoint = (
            fold_dir / f"k{fold}_cross_val_sub_{subfolds[fold]}"
            / f"best_model_ep_{epochs[fold]}.pt"
        )
        checkpoint.parent.mkdir(parents=True)
        checkpoint.touch()
        results[f"k{fold}_cross_val_final"] = {
            "best_model": {
                "best_model_idx": epochs[fold],
                "fold_sub_fold_idx": (fold, subfolds[fold]),
            }
        }
    video = videos / "subject" / "clip.mp4"
    video.parent.mkdir(parents=True)
    video.touch()
    payload = {
        "model_advanced_params": {
            "head": "ATTENTIVE_JEPA",
            "path_dataset": str(videos),
            "features_folder_saving_path": "UNBC/video/features/VideoMaev2_S",
            "model_type": "VIDEOMAE_v2_S",
        },
        "config": {},
        "results": results,
    }
    root.mkdir(exist_ok=True)
    with (root / "k_fold_results.pkl").open("wb") as stream:
        pickle.dump(payload, stream)
    return root


def _native_experiment_for_model(
        tmp_path, name, model, *, dataset="UNBC", target_max=4.0):
    root = _native_experiment(tmp_path / name)
    result_path = root / "k_fold_results.pkl"
    with result_path.open("rb") as stream:
        payload = pickle.load(stream)
    params = payload["model_advanced_params"]
    params["model_type"] = "VIDEOMAE_v2_S" if model == "VideoMAE" else "DFER"
    params["features_folder_saving_path"] = f"{dataset}/video/features/{model}"
    payload["config"]["max_label"] = target_max
    with result_path.open("wb") as stream:
        pickle.dump(payload, stream)
    return root


def test_native_fold_subfold_epoch_checkpoint_csv_row_and_video_resolution(tmp_path):
    root = _native_experiment(tmp_path)

    selected = pcp.resolve_native_experiment(root, "42")

    assert selected.fold == 1
    assert selected.subfold == 2
    assert selected.epoch == 7
    assert selected.checkpoint.name == "best_model_ep_7.pt"
    assert selected.test_csv.name == "test.csv"
    assert selected.row["sample_id"] == "42"
    assert selected.video_path == tmp_path / "videos" / "subject" / "clip.mp4"


def test_native_membership_must_be_unambiguous(tmp_path):
    root = _native_experiment(tmp_path, fold_samples=((42,), (42,)))
    with pytest.raises(ValueError, match="multiple test folds"):
        pcp.resolve_native_experiment(root, "42")


def test_native_comparison_selections_must_refer_to_the_same_sample():
    first = SimpleNamespace(row=_row(42, sample_name="clip-a"))
    second = SimpleNamespace(row=_row(42, sample_name="clip-b"))

    with pytest.raises(ValueError, match="same sample"):
        pcp.validate_native_comparison_selections(first, second)


def _model_checkpoint(root, side, index, sample_ids):
    model_root = root / f"{side}_model_{index}"
    checkpoint = (
        model_root / "train_ATTENTIVE_JEPA" / f"k{index}_cross_val"
        / f"k{index}_cross_val_sub_0" / "best_model_ep_3.pt"
    )
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    _write_tsv(checkpoint.parents[1] / "test.csv", [_row(sid) for sid in sample_ids])
    return checkpoint


def _cross_experiment(tmp_path, *, modes=("linear_only", "projector_linear"),
                      projector_kind="linear"):
    root = tmp_path / "cross"
    aggregate_dir = root / "aggregated_1"
    aggregate_dir.mkdir(parents=True)
    old_models = [
        _model_checkpoint(tmp_path, "old", 0, [10, 77]),
        _model_checkpoint(tmp_path, "old", 1, [42, 77]),
    ]
    new_models = [
        _model_checkpoint(tmp_path, "new", 0, [20, 77]),
        _model_checkpoint(tmp_path, "new", 1, [21]),
    ]
    subtrials = []
    subtrial_pkls = []
    for new_idx, new_model in enumerate(new_models):
        for old_idx, old_model in enumerate(old_models):
            trial_dir = root / f"trial_{new_idx}_{old_idx}"
            trial_dir.mkdir()
            projector = trial_dir / "projector.pt"
            projector.touch()
            refinements = {}
            for mode in modes:
                ref_dir = trial_dir / mode
                ref_dir.mkdir()
                linear = ref_dir / "linear_after.pt"
                linear.touch()
                project_after = ref_dir / "projector_after.pt"
                project_after.touch()
                refinements[mode] = {
                    "refine_mode": mode,
                    "linear_after_pth": str(linear),
                    "projector_after_pth": str(project_after),
                }
            payload = {
                "config_cross_space_projection": {
                    "new_model_pth": str(new_model),
                    "old_model_pth": str(old_model),
                    "interpolation_similarity": projector_kind,
                    "out_dir": str(trial_dir),
                },
                "linear_projector": {
                    "kind": projector_kind,
                    "ckpt_path": str(projector),
                    "config": {},
                    "norm_stats": None,
                },
                "refinements": refinements,
                "old_model_config": {
                    "model_advanced_params": {
                        "model_type": "VIDEOMAE_v2_S",
                        "features_folder_saving_path": (
                            "UNBC/video/features/VideoMaev2_S"
                        ),
                    },
                    "config": {},
                },
                "new_model_config": {
                    "model_advanced_params": {
                        "model_type": "DFER",
                        "features_folder_saving_path": (
                            "partA/video/features/DFER/Biovid"
                        ),
                    },
                    "config": {},
                },
            }
            result_path = trial_dir / "results_1.pkl"
            with result_path.open("wb") as stream:
                pickle.dump(payload, stream)
            subtrials.append({
                "new_idx": new_idx,
                "old_idx": old_idx,
                "new_model_pth": str(new_model),
                "old_model_pth": str(old_model),
            })
            subtrial_pkls.append(os.path.relpath(result_path, aggregate_dir))
    aggregate = {
        "aggregated": True,
        "subtrials": subtrials,
        "subtrial_pkls": subtrial_pkls,
    }
    with (aggregate_dir / "results_aggregate.pkl").open("wb") as stream:
        pickle.dump(aggregate, stream)
    return root


def test_cross_source_fixes_source_fold_and_selects_all_target_variants(tmp_path):
    root = _cross_experiment(tmp_path)

    selected = pcp.resolve_cross_experiment(root, "42", "source", 2)

    assert selected.fixed_index == 1
    assert [(item.new_idx, item.old_idx) for item in selected.variants] == [(0, 1), (1, 1)]
    assert all(item.refinement["refine_mode"] == "projector_linear"
               for item in selected.variants)


def test_cross_aggregate_loads_only_variants_for_the_selected_fold(tmp_path):
    root = _cross_experiment(tmp_path)
    loaded = []

    def loader(path):
        loaded.append(Path(path))
        with Path(path).open("rb") as stream:
            return pickle.load(stream)

    selected = pcp.resolve_cross_experiment(
        root, "42", "source", 2, load_pickle=loader)

    assert len(selected.variants) == 2
    assert len(loaded) == 3  # aggregate plus the two selected target variants


def test_cross_target_fixes_target_fold_and_selects_all_source_variants(tmp_path):
    root = _cross_experiment(tmp_path)

    selected = pcp.resolve_cross_experiment(root, "77", "target", 1)

    assert selected.fixed_index == 0
    assert [(item.new_idx, item.old_idx) for item in selected.variants] == [(0, 0), (0, 1)]
    assert all(item.refinement["refine_mode"] == "linear_only"
               for item in selected.variants)


def test_cross_membership_must_be_unambiguous(tmp_path):
    root = _cross_experiment(tmp_path)
    with pytest.raises(ValueError, match="multiple source test folds"):
        pcp.resolve_cross_experiment(root, "77", "source", 1)


def test_ambiguous_aggregate_pickles_are_rejected(tmp_path):
    root = _cross_experiment(tmp_path)
    second = root / "aggregated_2"
    second.mkdir()
    (second / "results_2.pkl").write_bytes(
        (root / "aggregated_1" / "results_aggregate.pkl").read_bytes())
    with pytest.raises(ValueError, match="ambiguous cross result pickles"):
        pcp.resolve_cross_experiment(root, "42", "source", 1)


def test_unsupported_projector_type_is_rejected(tmp_path):
    root = _cross_experiment(tmp_path, projector_kind="cos")
    with pytest.raises(ValueError, match="unsupported projector type.*cos"):
        pcp.resolve_cross_experiment(root, "42", "source", 1)


def test_missing_refinement_stage_lists_available_stages(tmp_path):
    root = _cross_experiment(tmp_path)
    with pytest.raises(
            ValueError,
            match=r"available stages: 1, 2.*linear_only.*projector_linear"):
        pcp.resolve_cross_experiment(root, "42", "source", 4)


def test_cuda_is_required():
    torch_without_cuda = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False))
    with pytest.raises(RuntimeError, match="CUDA is required"):
        pcp.require_cuda(torch_without_cuda)


def test_source_and_target_curve_sets_have_expected_labels_and_means():
    source = pcp.source_curve_set(
        [1.0, 2.0],
        {0: [2.0, 4.0], 1: [4.0, 8.0]},
        {0: [3.0, 6.0], 1: [5.0, 10.0]},
        stage=2,
        source_label="VideoMAE (UNBC)",
        target_label="DFER (BIOVID)",
    )
    assert list(source.summary) == [
        "Source model — VideoMAE (UNBC)",
        "Projection-only (mean) — VideoMAE (UNBC) → DFER (BIOVID)",
        "Task-aware ref. (mean) — VideoMAE (UNBC) → DFER (BIOVID)",
    ]
    assert np.allclose(
        source.summary[
            "Projection-only (mean) — VideoMAE (UNBC) → DFER (BIOVID)"
        ],
        [3.0, 6.0],
    )
    assert list(source.detail) == [
        "Target new_idx=0 — DFER (BIOVID)",
        "Target new_idx=1 — DFER (BIOVID)",
    ]

    stage_four = pcp.source_curve_set(
        [1.0], {0: [2.0]}, {0: [3.0]}, stage=4,
        source_label="VideoMAE (UNBC)", target_label="DFER (BIOVID)",
    )
    assert list(stage_four.summary)[-1] == (
        "Mean stage 4 refinement — VideoMAE (UNBC) → DFER (BIOVID)"
    )

    target = pcp.target_curve_set(
        [1.0, 2.0],
        {0: [2.0, 6.0], 1: [4.0, 8.0]},
        stage=1,
        source_label="VideoMAE (UNBC)",
        target_label="DFER (BIOVID)",
    )
    assert list(target.summary) == [
        "Native target — DFER (BIOVID)",
        "Mean refined linear (stage 1) — VideoMAE (UNBC) → DFER (BIOVID)",
    ]
    assert np.allclose(
        target.summary[
            "Mean refined linear (stage 1) — VideoMAE (UNBC) → DFER (BIOVID)"
        ],
        [3.0, 7.0],
    )
    assert list(target.detail) == [
        "Native target — DFER (BIOVID)",
        "Source old_idx=0 — VideoMAE (UNBC) → DFER (BIOVID)",
        "Source old_idx=1 — VideoMAE (UNBC) → DFER (BIOVID)",
    ]


def test_prediction_title_identifies_origin_and_cross_model_datasets():
    result = pcp.InferenceResult(
        curves=pcp.CurveSet(summary={}, detail={}),
        frame_count=1,
        ground_truth=2.0,
        metadata="source fold/old_idx 0, stage 2",
        origin_dataset="UNBC",
        origin_model="VideoMAE",
        source_dataset="UNBC",
        source_model="VideoMAE",
        target_dataset="BIOVID",
        target_model="DFER",
    )

    assert pcp.prediction_title("30", "source", result) == (
        "UNBC · Sample 30 · Origin source: VideoMAE\n"
        "Cross-space: VideoMAE (UNBC) → DFER (BIOVID) · "
        "source fold/old_idx 0, stage 2"
    )


def test_prediction_title_identifies_native_dataset_and_model():
    result = pcp.InferenceResult(
        curves=pcp.CurveSet(summary={}, detail={}),
        frame_count=1,
        ground_truth=2.0,
        metadata="native fold 0, subfold 0, epoch 46",
        origin_dataset="UNBC",
        origin_model="VideoMAE",
    )

    assert pcp.prediction_title("59", "native", result) == (
        "UNBC · Sample 59 · Native VideoMAE\n"
        "native fold 0, subfold 0, epoch 46"
    )


@pytest.mark.parametrize(
    ("origin", "origin_dataset", "origin_model", "stage", "expected"),
    [
        ("native", "BIOVID", "VideoMAE", None,
         "Sample 123 (BIOVID) · Origin-native: VideoMAE · labels range: 0-4"),
        ("native", "BIOVID", "VideoMAE_DFER", None,
         "Sample 123 (BIOVID) · Origin-native: VideoMAE vs DFER · labels range: 0-4"),
        ("source", "BIOVID", "VideoMAE", 1,
         "Sample 123 (BIOVID) · Origin-source: VideoMAE · labels range: 0-4\n"
         "Cross-space: VideoMAE (BIOVID) -> DFER (UNBC) · projector-only"),
        ("target", "UNBC", "DFER", 2,
         "Sample 123 (UNBC) · Origin-target: DFER · labels range: 0-4\n"
         "Cross-space: VideoMAE (BIOVID) -> DFER (UNBC) · task-aware refin."),
        ("source", "BIOVID", "VideoMAE", 4,
         "Sample 123 (BIOVID) · Origin-source: VideoMAE · labels range: 0-4\n"
         "Cross-space: VideoMAE (BIOVID) -> DFER (UNBC) · Stage 4 refin."),
    ],
)
def test_presentation_title_uses_origin_models_and_stage(
        origin, origin_dataset, origin_model, stage, expected):
    result = pcp.InferenceResult(
        curves=pcp.CurveSet(summary={}, detail={}),
        frame_count=16,
        ground_truth=2.0,
        metadata="fold 9, epoch 99",
        origin_dataset=origin_dataset,
        origin_model=origin_model,
        source_dataset="BIOVID",
        source_model="VideoMAE",
        target_dataset="UNBC",
        target_model="DFER",
    )

    assert pcp.presentation_title("123", origin, result, 4.0, stage=stage) == expected


def test_long_cross_presentation_title_fits_figure():
    import matplotlib.pyplot as plt

    result = pcp.InferenceResult(
        curves=pcp.CurveSet(summary={"Prediction": [1.0, 2.0]}, detail={}),
        frame_count=32, ground_truth=2.0, metadata="",
        origin_dataset="MINTPAIN", origin_model="VideoMAE",
        source_dataset="MINTPAIN", source_model="VideoMAE",
        target_dataset="BIOVID", target_model="DFER",
    )
    title = pcp.presentation_title("3122", "source", result, 4.0, stage=4)
    figure = pcp.make_prediction_figure(
        result.curves.summary, [(0, 15), (0, 31)], 2.0, title, y_max=4.0)
    try:
        figure.canvas.draw()
        bounds = figure.axes[0].title.get_window_extent(figure.canvas.get_renderer())
        assert bounds.x0 >= 0
        assert bounds.x1 <= figure.bbox.x1
    finally:
        plt.close(figure)


def _native_result(model, predictions, *, dataset="UNBC", frame_count=20,
                   ground_truth=3.0):
    curves = {f"Native {model} ({dataset})": predictions}
    return pcp.InferenceResult(
        curves=pcp.CurveSet(summary=curves, detail=curves),
        frame_count=frame_count,
        ground_truth=ground_truth,
        metadata="native metadata",
        origin_dataset=dataset,
        origin_model=model,
    )


def test_native_comparison_orders_models_and_uses_compact_full_range_title():
    comparison, axis_max = pcp.combine_native_results(
        _native_result("DFER", [2.0, 2.5]),
        _native_result("VideoMAE", [1.0, 1.5]),
        4.0,
        4.0,
    )

    assert list(comparison.curves.summary) == [
        "VideoMAE (native)", "DFER (native)",
    ]
    assert comparison.curves.summary["VideoMAE (native)"] == [1.0, 1.5]
    assert comparison.curves.summary["DFER (native)"] == [2.0, 2.5]
    assert comparison.curves.detail == comparison.curves.summary
    assert axis_max == 4.0
    assert pcp.native_comparison_title("59", comparison, axis_max) == (
        "VideoMAE vs DFER · UNBC · Sample 59 · Full range 0–4"
    )


@pytest.mark.parametrize(
    ("first", "second", "first_max", "second_max", "message"),
    [
        (
            _native_result("DFER", [1.0]),
            _native_result("DFER", [2.0]),
            4.0,
            4.0,
            "one VideoMAE and one DFER",
        ),
        (
            _native_result("DFER", [1.0]),
            _native_result("VideoMAE", [2.0], dataset="BIOVID"),
            4.0,
            4.0,
            "same dataset",
        ),
        (
            _native_result("DFER", [1.0], frame_count=16),
            _native_result("VideoMAE", [2.0], frame_count=17),
            4.0,
            4.0,
            "same frame count",
        ),
        (
            _native_result("DFER", [1.0], ground_truth=2.0),
            _native_result("VideoMAE", [2.0], ground_truth=3.0),
            4.0,
            4.0,
            "same ground truth",
        ),
        (
            _native_result("DFER", [1.0]),
            _native_result("VideoMAE", [2.0]),
            4.0,
            5.0,
            "same prediction range",
        ),
        (
            _native_result("DFER", [1.0]),
            _native_result("VideoMAE", [2.0, 3.0]),
            4.0,
            4.0,
            "same prefix count",
        ),
    ],
)
def test_native_comparison_rejects_incompatible_results(
        first, second, first_max, second_max, message):
    with pytest.raises(ValueError, match=message):
        pcp.combine_native_results(first, second, first_max, second_max)


def test_plot_suffix_stays_on_short_first_title_line():
    title = (
        "BIOVID · Sample 59 · Origin target: DFER\n"
        "Cross-space: VideoMAE (UNBC) → DFER (BIOVID) · fold metadata"
    )

    assert pcp.prediction_figure_title(title, "Per-model detail, stage 2") == (
        "BIOVID · Sample 59 · Origin target: DFER · Per-model detail, stage 2\n"
        "Cross-space: VideoMAE (UNBC) → DFER (BIOVID) · fold metadata"
    )


@pytest.mark.parametrize(
    ("debug", "title", "expected_titles"),
    [
        (False,
         "Sample 123 (BIOVID) · Origin-native: VideoMAE · labels range: 0-4",
         ["Sample 123 (BIOVID) · Origin-native: VideoMAE · labels range: 0-4"] * 2),
        (True,
         "BIOVID · Sample 123 · Native VideoMAE\nfold 1, epoch 7",
         [
             "BIOVID · Sample 123 · Native VideoMAE · Summary, full range 0–4\n"
             "fold 1, epoch 7",
             "BIOVID · Sample 123 · Native VideoMAE · Per-model detail, stage 2, "
             "full range 0–4\nfold 1, epoch 7",
         ]),
    ],
)
def test_video_title_mode_and_end_frame_axis(
        tmp_path, monkeypatch, debug, title, expected_titles):
    curves = pcp.CurveSet(summary={"Prediction": [1.0, 2.0, 3.0]},
                          detail={"Prediction": [1.0, 2.0, 3.0]})
    seen = []
    make_figure = pcp.make_prediction_figure

    def capture_figure(*args, **kwargs):
        figure = make_figure(*args, **kwargs)
        axis = figure.axes[0]
        seen.append((axis.get_title(), axis.get_xlabel(),
                     [tick.get_text() for tick in axis.get_xticklabels()]))
        return figure

    monkeypatch.setattr(pcp, "make_prediction_figure", capture_figure)
    monkeypatch.setattr(
        pcp, "_write_synchronized_videos", lambda *args, **kwargs: kwargs["paths"])
    pcp.save_prediction_videos(
        curves, [(0, 15), (0, 31), (0, 32)], 2.0,
        title,
        source_video=tmp_path / "unused.mp4", output_dir=tmp_path,
        stage=2, digest="abcdef123456", y_max=4.0, speed=1.0,
        debug_title=debug,
    )

    assert seen == [
        (expected_title, "End frame", ["15", "31", "32"])
        for expected_title in expected_titles
    ]


def test_target_origin_applies_refined_linears_directly_to_native_embeddings():
    embeddings = np.array([[1.0, 2.0], [3.0, 4.0]])
    seen = []

    def predict(linear, embedding, config):
        seen.append((linear, embedding.copy(), config))
        return float(embedding.sum() + linear)

    predictions = pcp.target_refined_predictions(
        embeddings,
        [(3, 10.0, {"name": "a"}), (4, 20.0, {"name": "b"})],
        predict_linear=predict,
    )

    assert predictions == {3: [13.0, 17.0], 4: [23.0, 27.0]}
    assert all(np.array_equal(call[1], embeddings[index % 2])
               for index, call in enumerate(seen))


def test_plot_contains_curves_ground_truth_ranges_metadata_and_grid(tmp_path):
    curves = {"Native target": [1.0, 2.0, 2.2], "Refined": [1.5, 2.5, 2.7]}
    figure = pcp.make_prediction_figure(
        curves,
        [(0, 15), (0, 31), (0, 32)],
        ground_truth=3.0,
        title="Sample 42 | target | fold 1 | stage 2",
    )
    axis = figure.axes[0]

    assert [line.get_label() for line in axis.lines] == [
        "Native target", "Refined", "Ground truth = 3"]
    assert "fold 1" in axis.get_title()
    assert [tick.get_text() for tick in axis.get_xticklabels()] == [
        "15", "31", "32"]
    assert axis.get_xlabel() == "End frame"
    assert all(tick.get_rotation() == 45 for tick in axis.get_xticklabels())
    assert all(tick.get_ha() == "right" for tick in axis.get_xticklabels())
    assert any(line.get_visible() for line in axis.get_xgridlines())

    paths = pcp.save_prediction_plots(
        pcp.CurveSet(summary=curves, detail=curves),
        [(0, 15), (0, 31), (0, 32)],
        3.0,
        "Sample 42 | target | fold 1 | stage 2",
        tmp_path,
        stage=2,
        digest="abcdef123456",
        y_max=5.0,
    )
    assert [path.name for path in paths] == [
        "summary_abcdef123456.png",
        "summary_full_range_abcdef123456.png",
        "models_stage_2_abcdef123456.png",
        "models_stage_2_full_range_abcdef123456.png",
    ]
    assert all(path.is_file() for path in paths)


def test_plot_can_use_full_dataset_y_range():
    figure = pcp.make_prediction_figure(
        {"Prediction": [1.0, 7.0]},
        [(0, 15), (0, 31)],
        ground_truth=3.0,
        title="Full range",
        y_max=5.0,
    )

    assert figure.axes[0].get_ylim() == (0.0, 5.0)


def test_extra_native_curve_is_rendered_only_in_full_range_summaries(tmp_path, monkeypatch):
    curves = pcp.CurveSet(
        summary={"Source": [1.0, 2.0]},
        detail={"Detail": [2.0, 3.0]},
        full_range_summary={"Source": [1.0, 2.0], "Target DFER": [1.5, 2.5]},
    )
    rendered = []
    real_figure = pcp.make_prediction_figure

    def capture_figure(data, *args, **kwargs):
        rendered.append(list(data))
        return real_figure(data, *args, **kwargs)

    monkeypatch.setattr(pcp, "make_prediction_figure", capture_figure)
    pcp.save_prediction_plots(
        curves, [(0, 15), (0, 31)], 3.0, "Sample", tmp_path,
        stage=2, digest="abcdef123456", y_max=4.0,
    )
    assert rendered == [
        ["Source"], ["Source", "Target DFER"], ["Detail"], ["Detail"],
    ]

    monkeypatch.setattr(
        pcp, "_write_synchronized_videos", lambda *args, **kwargs: kwargs["paths"],
    )
    rendered.clear()
    pcp.save_prediction_videos(
        curves, [(0, 15), (0, 31)], 3.0, "Sample",
        source_video=tmp_path / "video.mp4", output_dir=tmp_path,
        stage=2, digest="abcdef123456", y_max=4.0, speed=1.0,
    )
    assert rendered == [["Source", "Target DFER"], ["Detail"]]


def test_script_snapshot_contains_only_the_portable_invocation(tmp_path):
    argv = ["experiment root", "59", "--origin", "target"]

    path = pcp.save_script_snapshot(tmp_path, argv, digest="abcdef123456")

    command = shlex.join([
        "python3",
        "plot_cumulative_predictions.py",
        *argv,
    ])
    assert path.name == "plot_cumulative_predictions_abcdef123456.txt"
    assert path.read_text(encoding="utf-8") == f"{command}\n"


def test_cli_rejects_origin_for_native_and_requires_it_for_cross(tmp_path):
    native = _native_experiment(tmp_path)
    with pytest.raises(ValueError, match="--origin is only valid"):
        pcp.run_cli([str(native), "42", "--origin", "source"])

    cross = _cross_experiment(tmp_path)
    with pytest.raises(ValueError, match="--origin is required"):
        pcp.run_cli([str(cross), "42"])


def test_cli_rejects_cross_experiment_as_native_comparison_root(tmp_path):
    native = _native_experiment(tmp_path / "native-root")
    cross = _cross_experiment(tmp_path / "cross-root")

    with pytest.raises(ValueError, match="requires two native experiments"):
        pcp.run_cli([
            str(native), "42", "--compare-native-root", str(cross),
        ])


def test_cli_checks_cuda_before_unpickling_experiment_artifacts(tmp_path, monkeypatch):
    native = _native_experiment(tmp_path)
    monkeypatch.setattr(
        pcp, "resolve_native_experiment",
        lambda *args: pytest.fail("artifacts loaded before CUDA check"),
    )
    monkeypatch.setattr(
        pcp, "require_cuda",
        lambda: (_ for _ in ()).throw(RuntimeError("CUDA is required")),
    )
    with pytest.raises(RuntimeError, match="CUDA is required"):
        pcp.run_cli([str(native), "42"])


def test_mocked_native_cli_writes_hashed_artifacts_in_organized_path(
        tmp_path, monkeypatch):
    native = _native_experiment(tmp_path)
    monkeypatch.setattr(pcp, "require_cuda", lambda *_: None)

    def infer(selection, batch_size):
        assert selection.fold == 1
        assert batch_size == 3
        return pcp.InferenceResult(
            curves=pcp.CurveSet(
                summary={"Native": [1.0, 2.0]},
                detail={"Native": [1.0, 2.0]},
            ),
            frame_count=20,
            ground_truth=3.0,
            metadata="fold 1, subfold 2, epoch 7",
            origin_dataset="UNBC",
            origin_model="VideoMAE",
        )

    argv = [
        str(native), "42", "--output-root", str(tmp_path / "plots"),
        "--backbone-batch-size", "3",
    ]
    paths = pcp.run_cli(argv, native_inference=infer)
    digest = hashlib.sha256("\0".join(argv).encode()).hexdigest()[:12]

    assert [path.name for path in paths] == [
        f"summary_{digest}.png",
        f"summary_full_range_{digest}.png",
        f"models_stage_2_{digest}.png",
        f"models_stage_2_full_range_{digest}.png",
    ]
    assert paths[0].parent == (
        tmp_path / "plots" / "videomae_unbc" / "42" / "native"
    )
    assert (
        paths[0].parent / f"plot_cumulative_predictions_{digest}.txt"
    ).is_file()


@pytest.mark.parametrize(
    ("debug", "expected"),
    [
        (False, "Sample 42 (UNBC) · Origin-native: VideoMAE · labels range: 0-3"),
        (True, "UNBC · Sample 42 · Native VideoMAE · Summary\nfold 1, epoch 7"),
    ],
)
def test_cli_selects_presentation_or_legacy_titles(
        tmp_path, monkeypatch, debug, expected):
    native = _native_experiment(tmp_path)
    monkeypatch.setattr(pcp, "require_cuda", lambda: None)
    captured = []
    make_figure = pcp.make_prediction_figure

    def capture_figure(*args, **kwargs):
        figure = make_figure(*args, **kwargs)
        captured.append(figure.axes[0].get_title())
        return figure

    monkeypatch.setattr(pcp, "make_prediction_figure", capture_figure)
    result = pcp.InferenceResult(
        curves=pcp.CurveSet(summary={"Native": [1.0, 2.0]},
                            detail={"Native": [1.0, 2.0]}),
        frame_count=20, ground_truth=2.0, metadata="fold 1, epoch 7",
        origin_dataset="UNBC", origin_model="VideoMAE",
    )
    argv = [str(native), "42", "--output-root", str(tmp_path / "plots")]
    if debug:
        argv.append("--debug_title")

    pcp.run_cli(argv, native_inference=lambda *args: result)

    assert captured[0] == expected
    assert len(captured) == 4
    if not debug:
        assert captured == [expected] * 4


def test_native_comparison_cli_writes_one_full_range_plot_in_joint_directory(
        tmp_path, monkeypatch):
    dfer = _native_experiment_for_model(tmp_path, "dfer", "DFER")
    videomae = _native_experiment_for_model(tmp_path, "videomae", "VideoMAE")
    monkeypatch.setattr(pcp, "require_cuda", lambda *_: None)

    def infer(selection, batch_size):
        dataset, model = pcp.model_dataset_identity(selection.config)
        predictions = [1.0, 1.5] if model == "VideoMAE" else [2.0, 2.5]
        return _native_result(model, predictions, dataset=dataset)

    argv = [
        str(dfer), "42", "--compare-native-root", str(videomae),
        "--output-root", str(tmp_path / "plots"),
    ]
    paths = pcp.run_cli(argv, native_inference=infer)
    digest = pcp.invocation_digest(argv)

    assert paths == (
        tmp_path / "plots" / "videomae_dfer_unbc" / "42" / "native"
        / f"native_comparison_full_range_{digest}.png",
    )
    assert paths[0].is_file()
    assert list(paths[0].parent.glob("*.png")) == [paths[0]]
    assert (
        paths[0].parent / f"plot_cumulative_predictions_{digest}.txt"
    ).is_file()


def test_mocked_cross_cli_uses_origin_dataset_model_pair_and_stage(
        tmp_path, monkeypatch):
    cross = _cross_experiment(tmp_path, projector_kind="mlp")
    monkeypatch.setattr(pcp, "require_cuda", lambda *_: None)

    result = pcp.InferenceResult(
        curves=pcp.CurveSet(
            summary={"Native": [1.0]},
            detail={"Native": [1.0]},
        ),
        frame_count=16,
        ground_truth=3.0,
        metadata="target fold/new_idx 0, stage 1",
        origin_dataset="BIOVID",
        origin_model="DFER",
        source_dataset="UNBC",
        source_model="VideoMAE",
        target_dataset="BIOVID",
        target_model="DFER",
    )

    argv = [
        str(cross), "77", "--origin", "target", "--refinement-stage", "1",
        "--output-root", str(tmp_path / "plots"),
    ]
    paths = pcp.run_cli(
        argv, cross_inference=lambda selection, batch_size: result
    )
    digest = hashlib.sha256("\0".join(argv).encode()).hexdigest()[:12]

    assert paths[0].parent == (
        tmp_path / "plots" / "videomae_unbc_to_biovid_dfer"
        / "77" / "mlp" / "dfer_biovid" / "stage_1"
    )
    assert (
        paths[0].parent / f"plot_cumulative_predictions_{digest}.txt"
    ).is_file()


def test_source_cli_adds_source_dataset_dfer_from_its_test_fold(
        tmp_path, monkeypatch):
    cross = _cross_experiment(tmp_path / "cross-case")
    native = _native_experiment_for_model(tmp_path, "native-dfer", "DFER")
    monkeypatch.setattr(pcp, "require_cuda", lambda: None)
    captured = {}

    def infer_native(selection, batch_size):
        captured["native_fold"] = selection.fold
        return _native_result("DFER", [1.5, 2.5])

    def save_plots(curves, *args, **kwargs):
        captured["curves"] = curves
        return ()

    monkeypatch.setattr(pcp, "save_prediction_plots", save_plots)
    cross_result = pcp.InferenceResult(
        curves=pcp.CurveSet(
            summary={"Source": [1.0, 2.0]}, detail={"Detail": [1.0, 2.0]},
        ),
        frame_count=20, ground_truth=3.0, metadata="source fold 1",
        origin_dataset="UNBC", origin_model="VideoMAE",
        source_dataset="UNBC", source_model="VideoMAE",
        target_dataset="BIOVID", target_model="DFER",
    )

    pcp.run_cli(
        [str(cross), "42", "--origin", "source", "--target-native-root", str(native),
         "--output-root", str(tmp_path / "plots")],
        native_inference=infer_native,
        cross_inference=lambda selection, batch_size: cross_result,
    )

    assert captured["native_fold"] == 1
    assert captured["curves"].summary == {"Source": [1.0, 2.0]}
    assert captured["curves"].detail == {"Detail": [1.0, 2.0]}
    assert captured["curves"].full_range_summary == {
        "Source": [1.0, 2.0], "Target model — DFER (UNBC)": [1.5, 2.5],
    }


@pytest.mark.parametrize(
    ("origin", "sample_id", "dataset", "model", "expected_label"),
    [
        ("source", "42", "UNBC", "DFER", "Target model — DFER (UNBC)"),
        ("target", "77", "BIOVID", "VideoMAE",
         "Source model — VideoMAE (BIOVID)"),
    ],
)
def test_cross_comparison_adds_missing_native_only_to_summary_and_video(
        tmp_path, monkeypatch, origin, sample_id, dataset, model, expected_label):
    cross = _cross_experiment(tmp_path / "cross-case")
    native = _native_experiment_for_model(
        tmp_path, "other-native", model, dataset=dataset,
    )
    if origin == "target":
        _write_tsv(native / "train_ATTENTIVE_JEPA" / "k1_cross_val" / "test.csv",
                   [_row(77)])
    monkeypatch.setattr(pcp, "require_cuda", lambda: None)
    monkeypatch.setattr(pcp, "resolve_selection_video", lambda *_: tmp_path / "video.mp4")
    captured = {}
    monkeypatch.setattr(
        pcp, "save_prediction_plots",
        lambda curves, *args, **kwargs: captured.setdefault("plots", curves) and (),
    )
    monkeypatch.setattr(
        pcp, "save_prediction_videos",
        lambda curves, *args, **kwargs: captured.setdefault("videos", curves) and (),
    )
    summary = {
        "Native fixed": [1.0, 2.0],
        "Projected mean": [2.0, 3.0],
    }
    if origin == "source":
        summary["Refined mean"] = [2.5, 3.5]
    detail = {f"variant {index}": [float(index), float(index + 1)]
              for index in range(5)}
    result = pcp.InferenceResult(
        curves=pcp.CurveSet(summary=summary, detail=detail),
        frame_count=20, ground_truth=3.0, metadata="cross",
        origin_dataset=dataset,
        origin_model="VideoMAE" if origin == "source" else "DFER",
        source_dataset="UNBC", source_model="VideoMAE",
        target_dataset="BIOVID", target_model="DFER",
    )

    pcp.run_cli(
        [str(cross), sample_id, "--origin", origin,
         "--compare-native-root", str(native), "--video",
         "--output-root", str(tmp_path / "plots")],
        native_inference=lambda selection, batch_size: _native_result(
            model, [1.5, 2.5], dataset=dataset),
        cross_inference=lambda selection, batch_size: result,
    )

    curves = captured["plots"]
    assert curves.summary == {**summary, expected_label: [1.5, 2.5]}
    assert curves.full_range_summary == curves.summary
    assert curves.detail == detail
    assert captured["videos"] is curves


@pytest.mark.parametrize(
    ("dataset", "model"),
    [("BIOVID", "DFER"), ("UNBC", "VideoMAE")],
)
def test_source_cli_rejects_a_wrong_target_native_experiment(
        tmp_path, monkeypatch, dataset, model):
    cross = _cross_experiment(tmp_path / "cross-case")
    native = _native_experiment_for_model(
        tmp_path, "wrong-native", model, dataset=dataset,
    )
    monkeypatch.setattr(pcp, "require_cuda", lambda: None)
    cross_result = pcp.InferenceResult(
        curves=pcp.CurveSet(summary={"Source": [1.0, 2.0]}, detail={}),
        frame_count=20, ground_truth=3.0, metadata="source fold 1",
        origin_dataset="UNBC", origin_model="VideoMAE",
        source_dataset="UNBC", source_model="VideoMAE",
        target_dataset="BIOVID", target_model="DFER",
    )

    with pytest.raises(ValueError, match="source dataset and target model architecture"):
        pcp.run_cli(
            [str(cross), "42", "--origin", "source",
             "--target-native-root", str(native)],
            cross_inference=lambda selection, batch_size: cross_result,
            native_inference=lambda *args: pytest.fail("wrong model was inferred"),
        )


@pytest.mark.parametrize(
    ("frame_count", "ground_truth", "error"),
    [(21, 3.0, "different frame count"), (20, 4.0, "different ground truth")],
)
def test_source_cli_rejects_mismatched_target_native_results(
        tmp_path, monkeypatch, frame_count, ground_truth, error):
    cross = _cross_experiment(tmp_path / "cross-case")
    native = _native_experiment_for_model(tmp_path, "native-dfer", "DFER")
    monkeypatch.setattr(pcp, "require_cuda", lambda: None)
    cross_result = pcp.InferenceResult(
        curves=pcp.CurveSet(summary={"Source": [1.0, 2.0]}, detail={}),
        frame_count=20, ground_truth=3.0, metadata="source fold 1",
        origin_dataset="UNBC", origin_model="VideoMAE",
        source_dataset="UNBC", source_model="VideoMAE",
        target_dataset="BIOVID", target_model="DFER",
    )

    with pytest.raises(ValueError, match=error):
        pcp.run_cli(
            [str(cross), "42", "--origin", "source",
             "--target-native-root", str(native)],
            cross_inference=lambda selection, batch_size: cross_result,
            native_inference=lambda selection, batch_size: _native_result(
                "DFER", [1.5, 2.5], frame_count=frame_count,
                ground_truth=ground_truth,
            ),
        )


def test_decode_video_reads_every_frame_from_zero_and_converts_to_rgb(tmp_path):
    class Capture:
        def __init__(self):
            self.frames = [
                np.array([[[1, 2, 3]]], dtype=np.uint8),
                np.array([[[4, 5, 6]]], dtype=np.uint8),
            ]
            self.released = False

        def isOpened(self):
            return not self.released

        def read(self):
            return ((True, self.frames.pop(0)) if self.frames else (False, None))

        def release(self):
            self.released = True

    capture = Capture()
    fake_cv2 = SimpleNamespace(
        VideoCapture=lambda path: capture,
        COLOR_BGR2RGB=1,
        cvtColor=lambda frame, code: frame[..., ::-1],
    )

    frames = pcp.decode_video(tmp_path / "video.mp4", cv2_module=fake_cv2)

    assert frames[:, 0, 0].tolist() == [[3, 2, 1], [6, 5, 4]]
    assert capture.released


def test_head_prefix_inference_uses_one_more_chunk_each_step_and_returns_embeddings():
    torch = pytest.importorskip("torch")

    class Head:
        is_classification = False

        def __init__(self):
            self.token_lengths = []

        def __call__(self, x, key_padding_mask=None, return_video_emb=True):
            self.token_lengths.append(x.shape[1])
            pooled = x.mean(dim=1)
            return {"logits": pooled[:, :1], "embeddings": pooled}

    head = Head()
    features = torch.tensor([
        [[[[1.0]], [[3.0]]]],
        [[[[5.0]], [[7.0]]]],
    ]).reshape(2, 2, 1, 1, 1)

    predictions, embeddings = pcp.run_head_prefixes(
        head, features, {"config": {}}, torch_module=torch, device="cpu")

    assert head.token_lengths == [2, 4]
    assert predictions == [2.0, 4.0]
    assert np.allclose(np.asarray(embeddings).reshape(-1), [2.0, 4.0])


@pytest.mark.parametrize("kind", [
    "linear", "mlp", "autoencoder", "procrustes", "linear_close"])
def test_all_supported_projector_networks_map_source_to_target_dimensions(kind):
    torch = pytest.importorskip("torch")
    projector = pcp.build_projector_network(
        6, 4, kind, activation="gelu", num_layers=2, encoder_ratio=2,
        torch_module=torch,
    )
    assert tuple(projector(torch.ones(3, 6)).shape) == (3, 4)


def test_projector_normalization_is_applied_before_and_after_projection():
    torch = pytest.importorskip("torch")
    projector = torch.nn.Linear(2, 2)
    with torch.no_grad():
        projector.weight.copy_(torch.eye(2))
        projector.bias.zero_()
    stats = {
        "old_mean": np.array([1.0, 1.0]),
        "old_std": np.array([2.0, 4.0]),
        "new_mean": np.array([10.0, 20.0]),
        "new_std": np.array([2.0, 3.0]),
    }
    result = pcp.apply_projector(
        projector, np.array([3.0, 5.0]), stats,
        torch_module=torch, device="cpu")
    assert np.allclose(result, [12.0, 23.0])


def test_source_stage_uses_learned_projector_for_one_and_refined_for_two_and_four(tmp_path):
    root = _cross_experiment(
        tmp_path,
        modes=("linear_only", "projector_linear", "random_projector_linear"),
    )
    for stage, expected_parent in [(1, "trial_0_1"),
                                   (2, "projector_linear"),
                                   (4, "random_projector_linear")]:
        selection = pcp.resolve_cross_experiment(root, "42", "source", stage)
        path = pcp.selected_projector_path(selection.variants[0], stage)
        assert path.parent.name == expected_parent


def test_direct_cross_result_pickle_is_supported(tmp_path):
    root = _cross_experiment(tmp_path)
    direct = root / "trial_0_1" / "results_1.pkl"
    selected = pcp.resolve_cross_experiment(direct, "42", "source", 2)
    assert len(selected.variants) == 1
    assert selected.fixed_index == 1
    assert selected.fixed_model_pth.name == "best_model_ep_3.pt"


def test_cross_root_with_aggregate_and_direct_results_is_ambiguous(tmp_path):
    root = _cross_experiment(tmp_path)
    (root / "results_direct.pkl").write_bytes(
        (root / "trial_0_1" / "results_1.pkl").read_bytes())
    with pytest.raises(ValueError, match="ambiguous cross result pickles"):
        pcp.resolve_cross_experiment(root, "42", "source", 1)


def test_missing_selected_refinement_checkpoint_is_rejected_during_resolution(tmp_path):
    root = _cross_experiment(tmp_path)
    missing = root / "trial_0_1" / "linear_only" / "linear_after.pt"
    missing.unlink()
    with pytest.raises(FileNotFoundError, match="artifact unavailable"):
        pcp.resolve_cross_experiment(root, "42", "source", 1)


def test_native_inference_builds_one_native_curve_from_raw_video_prefixes(
        tmp_path, monkeypatch):
    selection = pcp.resolve_native_experiment(_native_experiment(tmp_path), "42")
    runtime = SimpleNamespace(config=selection.config, head=object())
    monkeypatch.setattr(pcp, "load_runtime_model", lambda *args, **kwargs: runtime)
    monkeypatch.setattr(pcp, "decode_video", lambda path: np.zeros((20, 2, 2, 3)))
    monkeypatch.setattr(
        pcp, "extract_clip_features",
        lambda runtime, chunks, batch_size: np.zeros((2, 1, 1, 1, 2)),
    )
    monkeypatch.setattr(
        pcp, "run_head_prefixes",
        lambda head, features, config: ([1.0, 2.0], [np.ones(2), np.ones(2)]),
    )

    result = pcp.infer_native(selection, 4)

    assert result.frame_count == 20
    assert result.ground_truth == 3.0
    assert result.curves.summary == {
        "Native VideoMAE (UNBC)": [1.0, 2.0],
    }
    assert (result.origin_dataset, result.origin_model) == ("UNBC", "VideoMAE")
    assert "fold 1" in result.metadata


def test_target_cross_inference_never_loads_or_applies_a_projector(
        tmp_path, monkeypatch):
    selection = pcp.resolve_cross_experiment(
        _cross_experiment(tmp_path), "77", "target", 1)
    runtime = SimpleNamespace(
        config={"config": {}},
        head=SimpleNamespace(linear="native-linear"),
    )
    monkeypatch.setattr(pcp, "load_runtime_model", lambda *args, **kwargs: runtime)
    monkeypatch.setattr(pcp, "resolve_selection_video", lambda *args: tmp_path / "v.mp4")
    monkeypatch.setattr(pcp, "decode_video", lambda path: np.zeros((20, 2, 2, 3)))
    monkeypatch.setattr(
        pcp, "extract_clip_features",
        lambda runtime, chunks, batch_size: np.zeros((2, 1, 1, 1, 2)),
    )
    embeddings = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
    monkeypatch.setattr(
        pcp, "run_head_prefixes",
        lambda head, features, config: ([1.0, 2.0], embeddings),
    )
    monkeypatch.setattr(
        pcp, "load_refined_linear",
        lambda base, variant: 10.0 + variant.old_idx,
    )
    monkeypatch.setattr(
        pcp, "predict_linear_value",
        lambda linear, embedding, config, reference_head: linear + embedding.sum(),
    )
    monkeypatch.setattr(
        pcp, "load_projector_for_variant",
        lambda *args, **kwargs: pytest.fail("target origin used a projector"),
    )

    result = pcp.infer_cross(selection, 4)

    assert list(result.curves.detail) == [
        "Native target — DFER (BIOVID)",
        "Source old_idx=0 — VideoMAE (UNBC) → DFER (BIOVID)",
        "Source old_idx=1 — VideoMAE (UNBC) → DFER (BIOVID)",
    ]
    assert result.curves.detail[
        "Source old_idx=0 — VideoMAE (UNBC) → DFER (BIOVID)"
    ] == [13.0, 17.0]
    assert (
        result.origin_dataset,
        result.origin_model,
        result.source_dataset,
        result.source_model,
        result.target_dataset,
        result.target_model,
    ) == ("BIOVID", "DFER", "UNBC", "VideoMAE", "BIOVID", "DFER")


def test_video_option_is_disabled_when_absent_and_accepts_an_optional_speed():
    parser = pcp.build_parser()

    disabled = parser.parse_args(["experiment", "42"])
    default = parser.parse_args(["experiment", "42", "--video"])
    half_speed = parser.parse_args(["experiment", "42", "--video", "0.5"])
    double_speed = parser.parse_args(["experiment", "42", "--video", "2"])

    assert disabled.video_speed is None
    assert default.video_speed == 1.0
    assert half_speed.video_speed == 0.5
    assert double_speed.video_speed == 2.0


def test_native_comparison_root_is_optional():
    parser = pcp.build_parser()

    ordinary = parser.parse_args(["experiment", "42"])
    comparison = parser.parse_args([
        "experiment", "42", "--compare-native-root", "other-experiment",
    ])

    assert ordinary.compare_native_root is None
    assert comparison.compare_native_root == Path("other-experiment")


@pytest.mark.parametrize("option", ["--video-speed", "--video_speed"])
def test_redundant_video_speed_options_are_rejected(option):
    with pytest.raises(SystemExit):
        pcp.build_parser().parse_args(["experiment", "42", option, "2"])


@pytest.mark.parametrize("speed", ["0", "-1", "nan", "inf"])
def test_video_speed_rejects_non_positive_or_non_finite_values(speed):
    with pytest.raises(SystemExit):
        pcp.build_parser().parse_args(["experiment", "42", "--video", speed])


@pytest.mark.parametrize("speed", [0.0, -1.0, np.nan, np.inf])
def test_save_prediction_videos_rejects_invalid_speed_before_opening_source(
        tmp_path, monkeypatch, speed):
    monkeypatch.setattr(
        pcp,
        "_source_frames_and_fps",
        lambda *args, **kwargs: pytest.fail("source video was opened"),
    )

    with pytest.raises(ValueError, match="video speed must be positive and finite"):
        pcp.save_prediction_videos(
            pcp.CurveSet(summary={"Summary": [1.0]}, detail={"Detail": [1.0]}),
            [(0, 15)],
            ground_truth=3.0,
            title="Invalid speed",
            source_video=tmp_path / "source.mp4",
            output_dir=tmp_path,
            stage=2,
            digest="abcdef123456",
            y_max=5.0,
            speed=speed,
        )


def test_cursor_positions_hold_first_chunk_then_use_actual_chunk_endpoints():
    positions = pcp.cursor_positions([(0, 15), (0, 31), (0, 32)], 33)

    assert positions[[0, 15, 16, 23, 31, 32]].tolist() == pytest.approx(
        [0.0, 0.0, 1 / 16, 0.5, 1.0, 2.0]
    )


def test_cursor_positions_interpolate_to_a_partial_final_chunk_endpoint():
    positions = pcp.cursor_positions([(0, 15), (0, 22)], 23)

    assert positions[[15, 16, 19, 22]].tolist() == pytest.approx(
        [0.0, 1 / 7, 4 / 7, 1.0]
    )


def test_cursor_is_red_in_the_bgr_composite():
    cv2 = pytest.importorskip("cv2")
    plot = np.zeros((650, 1200, 3), dtype=np.uint8)

    pcp._draw_cursor(plot, 0.0, (10, 50, 10, 100), [(0, 15)], cv2_module=cv2)
    composite = pcp._composite_frame(
        plot, np.zeros((6, 8, 3), dtype=np.uint8), 4.0, 4.0, 1.0,
        cv2_module=cv2)

    assert composite[30, 10].tolist() == [0, 0, 255]


def test_cursor_endpoints_match_rendered_chunk_coordinates():
    cv2 = pytest.importorskip("cv2")
    import matplotlib.pyplot as plt

    ranges = [(0, 15), (0, 31)]
    figure = pcp.make_prediction_figure(
        {"Prediction": [1.0, 2.0]}, ranges, 3.0, "Cursor coordinates", y_max=5.0)
    figure.set_size_inches(12, 6.5)
    figure.set_dpi(100)
    figure.canvas.draw()
    expected = [
        int(round(figure.axes[0].transData.transform((index, 0))[0]))
        for index in range(len(ranges))
    ]
    plt.close(figure)
    plot, axis_pixels = pcp._full_range_plot_image(
        {"Prediction": [1.0, 2.0]}, ranges, 3.0, "Cursor coordinates", 5.0)
    top, bottom, *_ = axis_pixels

    for position, x in enumerate(expected):
        rendered = plot.copy()
        pcp._draw_cursor(rendered, position, axis_pixels, ranges, cv2_module=cv2)
        assert rendered[(top + bottom) // 2, x].tolist() == [255, 0, 0]


def test_video_plot_pads_full_dataset_y_range(monkeypatch):
    captured = {}
    make_figure = pcp.make_prediction_figure

    def capture_figure(*args, **kwargs):
        figure = make_figure(*args, **kwargs)
        captured["axis"] = figure.axes[0]
        return figure

    monkeypatch.setattr(pcp, "make_prediction_figure", capture_figure)

    pcp._full_range_plot_image(
        {"Prediction": [0.0, 5.0]},
        [(0, 15), (0, 31)],
        3.0,
        "Padded video range",
        5.0,
    )

    assert captured["axis"].get_ylim() == (-0.25, 5.25)


def _write_synthetic_video(path, *, fps=4.0, frame_count=5):
    cv2 = pytest.importorskip("cv2")
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (8, 6), True)
    assert writer.isOpened()
    for index in range(frame_count):
        writer.write(np.full((6, 8, 3), index * 30, dtype=np.uint8))
    writer.release()


def test_synchronized_videos_preserve_frames_speed_and_composite_properties(tmp_path):
    cv2 = pytest.importorskip("cv2")
    source = tmp_path / "source.mp4"
    _write_synthetic_video(source)
    curves = pcp.CurveSet(
        summary={"Summary": [1.0, 2.0]}, detail={"Detail": [2.0, 3.0]})

    paths = pcp.save_prediction_videos(
        curves,
        [(0, 3), (0, 4)],
        ground_truth=3.0,
        title="Synthetic",
        source_video=source,
        output_dir=tmp_path,
        stage=2,
        digest="abcdef123456",
        y_max=5.0,
        speed=2.0,
    )

    assert [path.name for path in paths] == [
        "summary_full_range_video_abcdef123456.mp4",
        "models_stage_2_full_range_video_abcdef123456.mp4",
    ]
    for path in paths:
        capture = cv2.VideoCapture(str(path))
        assert capture.isOpened()
        assert capture.get(cv2.CAP_PROP_FRAME_COUNT) == pytest.approx(5)
        assert capture.get(cv2.CAP_PROP_FPS) == pytest.approx(8.0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        available, frame = capture.read()
        capture.release()
        assert available
        assert width > 1200 and width % 2 == 0
        assert height == 650
        assert tuple(frame[-1, -1]) == (0, 0, 0)


def test_native_comparison_video_writes_one_synchronized_full_range_file(tmp_path):
    cv2 = pytest.importorskip("cv2")
    source = tmp_path / "source.mp4"
    _write_synthetic_video(source)
    curves = pcp.CurveSet(
        summary={"VideoMAE (native)": [1.0, 2.0], "DFER (native)": [2.0, 3.0]},
        detail={},
    )

    path = pcp.save_native_comparison_video(
        curves,
        [(0, 3), (0, 4)],
        ground_truth=3.0,
        title="VideoMAE vs DFER · UNBC · Sample 42 · Full range 0–4",
        source_video=source,
        output_dir=tmp_path,
        digest="abcdef123456",
        y_max=4.0,
        speed=2.0,
    )

    assert path.name == "native_comparison_full_range_video_abcdef123456.mp4"
    assert list(tmp_path.glob("native_comparison*.mp4")) == [path]
    capture = cv2.VideoCapture(str(path))
    assert capture.isOpened()
    assert capture.get(cv2.CAP_PROP_FRAME_COUNT) == pytest.approx(5)
    assert capture.get(cv2.CAP_PROP_FPS) == pytest.approx(8.0)
    capture.release()


def test_invalid_temporary_videos_leave_existing_outputs_untouched(
        tmp_path, monkeypatch):
    class DiscardingWriter:
        def __init__(self, path, *args):
            self.path = Path(path)

        def isOpened(self):
            return True

        def write(self, frame):
            pass

        def release(self):
            self.path.write_bytes(b"not a video")

    class UnreadableCapture:
        def __init__(self, path):
            pass

        def isOpened(self):
            return False

        def release(self):
            pass

    fake_cv2 = SimpleNamespace(
        VideoWriter_fourcc=lambda *args: 0,
        VideoWriter=DiscardingWriter,
        VideoCapture=UnreadableCapture,
    )
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)
    monkeypatch.setattr(
        pcp, "_source_frames_and_fps",
        lambda *args, **kwargs: ([np.zeros((2, 2, 3), dtype=np.uint8)], 4.0),
    )
    monkeypatch.setattr(
        pcp, "_full_range_plot_image",
        lambda *args: (np.zeros((2, 4, 3), dtype=np.uint8), (0, 1, 0, 3)),
    )
    monkeypatch.setattr(
        pcp, "_composite_frame",
        lambda *args, **kwargs: np.zeros((2, 4, 3), dtype=np.uint8),
    )
    monkeypatch.setattr(pcp, "_draw_cursor", lambda *args, **kwargs: None)
    paths = (
        tmp_path / "summary_full_range_video_abcdef123456.mp4",
        tmp_path / "models_stage_2_full_range_video_abcdef123456.mp4",
    )
    for path in paths:
        path.write_bytes(b"known-good-output")

    with pytest.raises(RuntimeError, match="encoded video validation"):
        pcp.save_prediction_videos(
            pcp.CurveSet(summary={"Summary": [1.0]}, detail={"Detail": [1.0]}),
            [(0, 0)], 3.0, "Discarded writes", source_video=tmp_path / "source.mp4",
            output_dir=tmp_path, stage=2, digest="abcdef123456", y_max=5.0, speed=1.0)

    assert [path.read_bytes() for path in paths] == [b"known-good-output"] * 2
    assert not list(tmp_path.glob(".*.tmp.mp4"))


def test_second_writer_construction_failure_releases_first_and_cleans_temp(
        tmp_path, monkeypatch):
    writers = []

    class Writer:
        def __init__(self, path):
            self.path = Path(path)
            self.released = False
            self.path.write_bytes(b"partial")

        def isOpened(self):
            return True

        def release(self):
            self.released = True

    def video_writer(path, *args):
        if writers:
            raise RuntimeError("second writer construction failed")
        writer = Writer(path)
        writers.append(writer)
        return writer

    monkeypatch.setitem(sys.modules, "cv2", SimpleNamespace(
        VideoWriter_fourcc=lambda *args: 0,
        VideoWriter=video_writer,
    ))
    monkeypatch.setattr(
        pcp, "_source_frames_and_fps",
        lambda *args, **kwargs: ([np.zeros((2, 2, 3), dtype=np.uint8)], 4.0),
    )
    monkeypatch.setattr(
        pcp, "_full_range_plot_image",
        lambda *args: (np.zeros((2, 4, 3), dtype=np.uint8), (0, 1, 0, 3)),
    )
    monkeypatch.setattr(
        pcp, "_composite_frame",
        lambda *args, **kwargs: np.zeros((2, 4, 3), dtype=np.uint8),
    )

    with pytest.raises(RuntimeError, match="second writer construction failed"):
        pcp.save_prediction_videos(
            pcp.CurveSet(summary={"Summary": [1.0]}, detail={"Detail": [1.0]}),
            [(0, 0)], 3.0, "Writer construction", source_video=tmp_path / "source.mp4",
            output_dir=tmp_path, stage=2, digest="abcdef123456", y_max=5.0, speed=1.0)

    assert writers[0].released
    assert not list(tmp_path.glob(".*.tmp.mp4"))


@pytest.mark.parametrize(
    ("video_args", "speed"),
    [(["--video"], 1.0), (["--video", "0.5"], 0.5), (["--video", "2"], 2.0)],
)
def test_native_video_cli_uses_selected_source_and_appends_videos(
        tmp_path, monkeypatch, video_args, speed):
    native = _native_experiment(tmp_path)
    source = tmp_path / "videos" / "subject" / "clip.mp4"
    seen = {}
    monkeypatch.setattr(pcp, "require_cuda", lambda *_: None)
    def save_videos(*args, source_video, output_dir, stage, digest, speed, **kwargs):
        seen.update(source_video=source_video, speed=speed)
        return (
            Path(output_dir) / f"summary_full_range_video_{digest}.mp4",
            Path(output_dir) / f"models_stage_{stage}_full_range_video_{digest}.mp4",
        )
    monkeypatch.setattr(pcp, "save_prediction_videos", save_videos)
    result = pcp.InferenceResult(
        curves=pcp.CurveSet(summary={"Native": [1.0]}, detail={"Native": [1.0]}),
        frame_count=16, ground_truth=3.0, metadata="native",
        origin_dataset="UNBC", origin_model="VideoMAE")
    argv = [
        str(native), "42", "--output-root", str(tmp_path / "plots"), *video_args,
    ]

    paths = pcp.run_cli(argv, native_inference=lambda *args: result)

    assert len(paths) == 6
    assert seen == {"source_video": source, "speed": speed}
    assert paths[-2:] == (
        paths[0].parent / f"summary_full_range_video_{pcp.invocation_digest(argv)}.mp4",
        paths[0].parent / f"models_stage_2_full_range_video_{pcp.invocation_digest(argv)}.mp4",
    )


@pytest.mark.parametrize(
    ("debug", "expected_title"),
    [
        (False, "Sample 42 (UNBC) · Origin-native: VideoMAE vs DFER · labels range: 0-4"),
        (True, "VideoMAE vs DFER · UNBC · Sample 42 · Full range 0–4"),
    ],
)
def test_native_comparison_video_cli_uses_first_source_and_appends_one_video(
        tmp_path, monkeypatch, debug, expected_title):
    videomae = _native_experiment_for_model(tmp_path, "videomae", "VideoMAE")
    dfer = _native_experiment_for_model(tmp_path, "dfer", "DFER")
    source = tmp_path / "videomae" / "videos" / "subject" / "clip.mp4"
    seen = {}
    monkeypatch.setattr(pcp, "require_cuda", lambda *_: None)

    def infer(selection, batch_size):
        dataset, model = pcp.model_dataset_identity(selection.config)
        return _native_result(model, [1.0, 2.0], dataset=dataset)

    def save_plot(*args, digest, **kwargs):
        return Path(args[4]) / f"native_comparison_full_range_{digest}.png"

    def save_video(*args, source_video, output_dir, digest, speed, **kwargs):
        seen.update(source_video=source_video, speed=speed, title=args[3])
        return Path(output_dir) / f"native_comparison_full_range_video_{digest}.mp4"

    monkeypatch.setattr(pcp, "save_native_comparison_plot", save_plot)
    monkeypatch.setattr(pcp, "save_native_comparison_video", save_video)
    argv = [
        str(videomae), "42", "--compare-native-root", str(dfer),
        "--video", "0.5", "--output-root", str(tmp_path / "plots"),
    ]
    if debug:
        argv.append("--debug_title")

    paths = pcp.run_cli(argv, native_inference=infer)
    digest = pcp.invocation_digest(argv)

    assert [path.name for path in paths] == [
        f"native_comparison_full_range_{digest}.png",
        f"native_comparison_full_range_video_{digest}.mp4",
    ]
    assert seen == {
        "source_video": source,
        "speed": 0.5,
        "title": expected_title,
    }


@pytest.mark.parametrize(("origin", "sample_id"), [("source", "42"), ("target", "77")])
def test_cross_video_cli_uses_origin_specific_source_path(
        tmp_path, monkeypatch, origin, sample_id):
    cross = _cross_experiment(tmp_path)
    source_root = tmp_path / "source-videos"
    target_root = tmp_path / "target-videos"
    for result_path in cross.glob("trial_*_*/results_1.pkl"):
        with result_path.open("rb") as stream:
            result = pickle.load(stream)
        result["old_model_config"]["model_advanced_params"]["path_dataset"] = str(
            source_root)
        result["new_model_config"]["model_advanced_params"]["path_dataset"] = str(
            target_root)
        with result_path.open("wb") as stream:
            pickle.dump(result, stream)
    expected_source = (
        source_root if origin == "source" else target_root
    ) / "subject" / "clip.mp4"
    for root in (source_root, target_root):
        video = root / "subject" / "clip.mp4"
        video.parent.mkdir(parents=True, exist_ok=True)
        video.touch()
    monkeypatch.setattr(pcp, "require_cuda", lambda *_: None)
    seen = {}
    monkeypatch.setattr(
        pcp, "save_prediction_videos",
        lambda *args, source_video, **kwargs: (
            seen.update(source_video=source_video) or
            (tmp_path / "summary.mp4", tmp_path / "detail.mp4")
        ),
    )
    result = pcp.InferenceResult(
        curves=pcp.CurveSet(summary={"Native": [1.0]}, detail={"Native": [1.0]}),
        frame_count=16, ground_truth=3.0, metadata="target",
        origin_dataset="BIOVID", origin_model="DFER", source_dataset="UNBC",
        source_model="VideoMAE", target_dataset="BIOVID", target_model="DFER")

    paths = pcp.run_cli(
        [str(cross), sample_id, "--origin", origin, "--video",
         "--output-root", str(tmp_path / "plots")],
        cross_inference=lambda *args: result,
    )

    assert seen == {"source_video": expected_source.resolve()}
    assert paths[-2:] == (tmp_path / "summary.mp4", tmp_path / "detail.mp4")
