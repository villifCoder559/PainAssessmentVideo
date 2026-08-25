import runpy
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd
import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


# custom.helper starts a multiprocessing Manager while cross_space_projection is
# imported. Anchor-selection tests do not use that profiler.
with mock.patch("multiprocessing.Manager") as manager:
    manager.return_value.dict.return_value = {}
    import cross_space_projection as csp


def test_balance_class_quality_selects_lowest_error_anchor_from_each_class():
    candidates = pd.DataFrame(
        {
            "sample_id": [1, 2, 3, 4, 5, 6],
            "class_id": [0, 0, 0, 1, 1, 1],
        }
    )
    quality_errors = {1: 0.8, 2: 0.1, 3: 0.4, 4: 0.5, 5: 0.2, 6: 0.9}

    selected = csp._select_anchors(
        candidates,
        num_anchors=2,
        selection_type="balance_class_quality",
        quality_scores=quality_errors,
    )

    assert selected["sample_id"].tolist() == [2, 5]


def test_balance_class_quality_covers_every_class_when_budget_is_smaller():
    candidates = pd.DataFrame(
        {
            "sample_id": [1, 2, 3, 4, 5, 6],
            "class_id": [0, 0, 1, 1, 2, 2],
        }
    )
    quality_errors = {1: 0.4, 2: 0.1, 3: 0.2, 4: 0.3, 5: 0.8, 6: 0.5}

    selected = csp._select_anchors(
        candidates,
        num_anchors=2,
        selection_type="balance_class_quality",
        quality_scores=quality_errors,
    )

    assert selected["sample_id"].tolist() == [2, 3, 6]


def test_balance_class_quality_redistributes_budget_from_an_exhausted_class():
    candidates = pd.DataFrame(
        {
            "sample_id": [1, 2, 3, 4, 5, 6],
            "class_id": [0, 1, 1, 1, 1, 1],
        }
    )
    quality_errors = {1: 0.6, 2: 0.5, 3: 0.1, 4: 0.4, 5: 0.2, 6: 0.3}

    selected = csp._select_anchors(
        candidates,
        num_anchors=4,
        selection_type="balance_class_quality",
        quality_scores=quality_errors,
    )

    assert selected["sample_id"].tolist() == [1, 3, 5, 6]


def test_balance_class_quality_breaks_equal_error_ties_by_sample_id():
    candidates = pd.DataFrame(
        {"sample_id": [30, 10, 20], "class_id": [0, 0, 1]}
    )

    selected = csp._select_anchors(
        candidates,
        num_anchors=2,
        selection_type="balance_class_quality",
        quality_scores={30: 0.1, 10: 0.1, 20: 0.2},
    )

    assert selected["sample_id"].tolist() == [10, 20]


def test_anchor_quality_scores_are_absolute_new_model_errors_keyed_by_sample_id():
    inference = {
        "sample_ids": np.array([30, 10, 20], dtype=np.int64),
        "labels": np.array([4.0, 1.0, 2.0], dtype=np.float32),
        "predictions": np.array([3.25, 1.5, 2.0], dtype=np.float32),
        "embeddings": np.zeros((3, 2), dtype=np.float32),
    }

    scores = csp._anchor_quality_scores(inference)

    assert scores == {30: 0.75, 10: 0.5, 20: 0.0}


def test_anchor_quality_scores_reject_non_scalar_predictions():
    inference = {
        "sample_ids": np.array([10, 20], dtype=np.int64),
        "labels": np.array([1.0, 2.0], dtype=np.float32),
        "predictions": np.array([[0.5, 0.5], [1.5, 1.5]], dtype=np.float32),
    }

    with pytest.raises(ValueError, match="one scalar prediction per candidate anchor"):
        csp._anchor_quality_scores(inference)


def test_anchor_quality_scores_reject_duplicate_sample_ids():
    inference = {
        "sample_ids": np.array([10, 10], dtype=np.int64),
        "labels": np.array([1.0, 2.0], dtype=np.float32),
        "predictions": np.array([0.5, 1.5], dtype=np.float32),
    }

    with pytest.raises(ValueError, match="duplicate sample_id.*10"):
        csp._anchor_quality_scores(inference)


def test_anchor_quality_scores_reject_non_finite_values():
    inference = {
        "sample_ids": np.array([10, 20], dtype=np.int64),
        "labels": np.array([1.0, 2.0], dtype=np.float32),
        "predictions": np.array([0.5, np.nan], dtype=np.float32),
    }

    with pytest.raises(ValueError, match="non-finite label or prediction.*20"):
        csp._anchor_quality_scores(inference)


def test_anchor_quality_scores_reject_mismatched_inference_lengths():
    inference = {
        "sample_ids": np.array([10, 20], dtype=np.int64),
        "labels": np.array([1.0, 2.0], dtype=np.float32),
        "predictions": np.array([0.5], dtype=np.float32),
    }

    with pytest.raises(ValueError, match="matching numbers of sample_ids, labels, and predictions"):
        csp._anchor_quality_scores(inference)


def test_balance_class_quality_requires_a_score_for_every_candidate():
    candidates = pd.DataFrame(
        {"sample_id": [10, 20, 30], "class_id": [0, 0, 1]}
    )

    with pytest.raises(ValueError, match="missing quality scores.*30"):
        csp._select_anchors(
            candidates,
            num_anchors=2,
            selection_type="balance_class_quality",
            quality_scores={10: 0.2, 20: 0.1},
        )


def test_balance_class_quality_rejects_non_finite_quality_scores():
    candidates = pd.DataFrame(
        {"sample_id": [10, 20], "class_id": [0, 1]}
    )

    with pytest.raises(ValueError, match="non-finite quality scores.*20"):
        csp._select_anchors(
            candidates,
            num_anchors=2,
            selection_type="balance_class_quality",
            quality_scores={10: 0.2, 20: np.inf},
        )


def test_extract_anchor_quality_pool_runs_new_model_on_every_candidate(tmp_path):
    candidates = pd.DataFrame(
        {"sample_id": [30, 10, 20], "class_id": [1, 0, 0]}
    )
    new_model = object()
    new_config = {"config": {"normalize_labels": 1, "max_label": 4}}
    expected_inference = {
        "sample_ids": np.array([10, 20, 30], dtype=np.int64),
        "labels": np.array([1.0, 2.0, 4.0], dtype=np.float32),
        "predictions": np.array([1.5, 2.0, 3.25], dtype=np.float32),
        "embeddings": np.arange(6, dtype=np.float32).reshape(3, 2),
    }

    def fake_extract(model, checkpoint, csv_path, config):
        assert model is new_model
        assert checkpoint == "new-model.pt"
        assert config is new_config
        assert sorted(pd.read_csv(csv_path, sep="\t")["sample_id"].tolist()) == [10, 20, 30]
        return expected_inference

    with mock.patch.object(csp, "_extract_embeddings", side_effect=fake_extract):
        inference, scores = csp._extract_anchor_quality_pool(
            new_model,
            "new-model.pt",
            new_config,
            candidates,
            tmp_path / "quality_candidates.csv",
        )

    assert inference is expected_inference
    assert scores == {10: 0.5, 20: 0.0, 30: 0.75}


def test_grid_precompute_caches_full_quality_inference_and_reuses_selected_embeddings(tmp_path):
    source_csv = tmp_path / "train.csv"
    pd.DataFrame(
        {
            "sample_id": [1, 2, 3, 4, 5, 6],
            "class_id": [0, 0, 0, 1, 1, 1],
        }
    ).to_csv(source_csv, index=False, sep="\t")
    new_model = object()
    old_model = object()
    new_calls = []

    def fake_extract(model, checkpoint, csv_path, config, features_path_override=None):
        frame = pd.read_csv(csv_path, sep="\t")
        sample_ids = frame["sample_id"].to_numpy(dtype=np.int64)
        if model is new_model:
            new_calls.append(sample_ids.tolist())
            assert features_path_override is None
            errors = {1: 0.8, 2: 0.1, 3: 0.4, 4: 0.5, 5: 0.2, 6: 0.9}
            labels = np.zeros(len(sample_ids), dtype=np.float32)
            predictions = np.array([errors[int(sid)] for sid in sample_ids], dtype=np.float32)
        else:
            assert model is old_model
            assert features_path_override == "old-backbone-new-domain"
            labels = np.zeros(len(sample_ids), dtype=np.float32)
            predictions = np.zeros(len(sample_ids), dtype=np.float32)
        return {
            "sample_ids": sample_ids,
            "labels": labels,
            "predictions": predictions,
            "embeddings": np.column_stack((sample_ids, sample_ids + 100)).astype(np.float32),
        }

    args = SimpleNamespace(
        projector_recipes=[csp.LINEAR_PROJECTOR_CONFIG],
        refinement_recipes=[csp.REFINEMENT_CONFIG],
        interpolation_similarity=["cos"],
        mlp_activation=["gelu"],
        mlp_num_layers=[1],
        refinement=0,
        csv_anchor_selection=["train"],
        num_anchors=[2, 4],
        anchor_selection_type=["balance_class_quality"],
        old_model_csv=[],
        rbf_sigma=[1.0],
    )

    with (
        mock.patch.object(csp, "_detect_dataset", return_value="NEW"),
        mock.patch.object(csp, "_detect_backbone", return_value="BACKBONE"),
        mock.patch.object(csp, "_get_features_path", return_value="old-backbone-new-domain"),
        mock.patch.object(csp, "_resolve_anchor_csvs", return_value=[str(source_csv)]),
        mock.patch.object(csp, "clean_csv_from_augmentations", side_effect=lambda path: path),
        mock.patch.object(csp.helper, "set_step_shift"),
        mock.patch.object(csp, "_extract_embeddings", side_effect=fake_extract),
    ):
        anchor_cache, _ = csp._precompute_embeddings(
            old_model,
            new_model,
            "old-model.pt",
            "new-model.pt",
            {"config": {}},
            {"config": {}},
            "old-features",
            "new-features",
            args,
            str(tmp_path),
        )

    assert new_calls == [[1, 2, 3, 4, 5, 6]]
    assert anchor_cache[("train", 2, "balance_class_quality")]["new"]["sample_ids"].tolist() == [2, 5]
    assert anchor_cache[("train", 4, "balance_class_quality")]["new"]["sample_ids"].tolist() == [2, 3, 5, 4]


def test_single_run_scores_full_pool_and_reuses_new_model_embeddings(tmp_path):
    anchor_csv = tmp_path / "train.csv"
    pd.DataFrame(
        {
            "sample_id": [1, 2, 3, 4, 5, 6],
            "class_id": [0, 0, 0, 1, 1, 1],
        }
    ).to_csv(anchor_csv, index=False, sep="\t")
    old_csv = tmp_path / "old_test.csv"
    pd.DataFrame(
        {"sample_id": [100, 200], "class_id": [0, 1]}
    ).to_csv(old_csv, index=False, sep="\t")

    class FakeNewModel:
        def __init__(self):
            self.head = torch.nn.Module()
            self.head.linear = torch.nn.Linear(2, 1, bias=False)
            torch.nn.init.zeros_(self.head.linear.weight)

    old_model = object()
    new_model = FakeNewModel()
    model_type = SimpleNamespace(name="FAKE")
    old_config = {
        "model_advanced_params": {
            "model_type": model_type,
            "features_folder_saving_path": "old-features",
        },
        "config": {},
    }
    new_config = {
        "model_advanced_params": {
            "model_type": model_type,
            "features_folder_saving_path": "new-features",
        },
        "config": {"normalize_labels": 0},
    }
    new_calls = []

    def fake_extract(model, checkpoint, csv_path, config, features_path_override=None):
        frame = pd.read_csv(csv_path, sep="\t")
        sample_ids = frame["sample_id"].to_numpy(dtype=np.int64)
        if model is new_model:
            new_calls.append(sample_ids.tolist())
            errors = {1: 0.8, 2: 0.1, 3: 0.4, 4: 0.5, 5: 0.2, 6: 0.9}
            labels = np.zeros(len(sample_ids), dtype=np.float32)
            predictions = np.array([errors[int(sid)] for sid in sample_ids], dtype=np.float32)
        elif features_path_override is not None:
            labels = np.zeros(len(sample_ids), dtype=np.float32)
            predictions = np.zeros(len(sample_ids), dtype=np.float32)
        else:
            labels = np.array([0.0, 1.0], dtype=np.float32)
            predictions = np.zeros(2, dtype=np.float32)
        return {
            "sample_ids": sample_ids,
            "labels": labels,
            "predictions": predictions,
            "embeddings": np.column_stack((sample_ids, sample_ids + 100)).astype(np.float32),
        }

    args = SimpleNamespace(
        old_model_pth="old-model.pt",
        new_model_pth="new-model.pt",
        num_anchors=2,
        anchor_selection_type="balance_class_quality",
        csv_anchor_selection="train",
        old_model_csv="test",
        interpolation_similarity="cos",
        mlp_activation="gelu",
        mlp_num_layers=1,
        weighting_method="rbf",
        rbf_sigma=1.0,
        run_tag="quality-test",
        fake_projection=False,
        fake_projection_distribution="matched_gaussian",
        refinement=0,
        projector_recipes=[csp.LINEAR_PROJECTOR_CONFIG],
        refinement_recipes=[csp.REFINEMENT_CONFIG],
    )

    with (
        mock.patch.object(
            csp,
            "_load_config",
            side_effect=lambda checkpoint: old_config if checkpoint == "old-model.pt" else new_config,
        ),
        mock.patch.object(csp, "_build_model", side_effect=[old_model, new_model]),
        mock.patch.object(csp, "_resolve_old_model_csvs", return_value=[str(old_csv)]),
        mock.patch.object(csp, "_resolve_anchor_csvs", return_value=[str(anchor_csv)]),
        mock.patch.object(csp, "clean_csv_from_augmentations", side_effect=lambda path: path),
        mock.patch.object(csp.helper, "set_step_shift"),
        mock.patch.object(csp, "_detect_dataset", return_value="NEW"),
        mock.patch.object(csp, "_detect_backbone", return_value="BACKBONE"),
        mock.patch.object(csp, "_get_features_path", return_value="old-backbone-new-domain"),
        mock.patch.object(csp, "_extract_embeddings", side_effect=fake_extract),
        mock.patch.object(
            csp,
            "_compute_weights",
            return_value=np.eye(2, dtype=np.float32),
        ),
    ):
        result_path = csp.cross_space_projection(args, out_root=tmp_path)

    result = pd.read_pickle(result_path)
    assert new_calls == [[1, 2, 3, 4, 5, 6]]
    assert result["new_model_anchors_embeddings"]["sample_ids"].tolist() == [2, 5]


def test_yaml_entrypoint_accepts_balance_class_quality(tmp_path, monkeypatch):
    config_path = tmp_path / "quality.yaml"
    config_path.write_text(
        "\n".join(
            [
                "new_model_pth: [/missing/new/a/b/c/model.pt]",
                "old_model_pth: [/missing/old/a/b/c/model.pt]",
                "num_anchors: [100]",
                "anchor_selection_type: [balance_class_quality]",
                "csv_anchor_selection: [train]",
                "old_model_csv: [test]",
                "interpolation_similarity: [linear]",
                "weighting_method: [none]",
                "rbf_sigma: [1.0]",
                "n_trials: null",
                "optuna_sampler: grid",
                "refinement: 0",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [str(REPO_ROOT / "cross_space_projection.py"), "--config", str(config_path)],
    )

    with pytest.raises(FileNotFoundError, match="k_fold_results.pkl"):
        runpy.run_path(str(REPO_ROOT / "cross_space_projection.py"), run_name="__main__")
